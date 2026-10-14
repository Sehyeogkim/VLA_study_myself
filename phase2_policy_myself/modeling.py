'''
modeling.py — 정책 모델 (B: VLM freeze + cross-attention action expert)

Full_script_train.py 에서 클래스만 그대로 옮김 (속성 이름 그대로 -> 기존 best.pt 로드 가능)
  - Prefix       : image + task + state -> VLM(text_model) 입력/출력
  - ActionExpert : x_t, t, VLM 출력 -> v_t
  - compute_loss : 학습/val 용 flow matching loss            (train.py 에서 사용)
  - sample_actions : noise -> Euler num_steps -> action chunk (eval.py 에서 사용)
  - load_policy  : 체크포인트 -> prefix, expert, cfg
'''
import torch
import torch.nn as nn
import math

from configuration import MyPolicyConfig


class Prefix(nn.Module):

    def __init__(self, cfg: MyPolicyConfig, state_mean, state_std):
        super().__init__()
        self.cfg = cfg

        from transformers import AutoModelForImageTextToText, AutoTokenizer
        self.vlm_model = AutoModelForImageTextToText.from_pretrained(cfg.vlm_name)
        self.vlm_model.eval()                    # freeze: dropout 끄기
        self.vlm_model.requires_grad_(False)     # freeze: 학습 안 함
    
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.vlm_name)

        # All the projection layers should be here.
        self.state_proj = nn.Linear(cfg.state_dim, cfg.vlm_hidden)

        # state normalize 값: buffer 로 등록 -> .to(device) 따라가고, state_dict(체크포인트)에도 저장됨
        self.register_buffer("state_mean", torch.as_tensor(state_mean, dtype=torch.float32))
        self.register_buffer("state_std", torch.as_tensor(state_std, dtype=torch.float32))

    def prefix_preprocessing(self, batch):

        # 1.  image: [B, 3, 96, 96] (0~1, 이미 CHW) -> 512 resize -> [-1, 1]
        import torch.nn.functional as F
        img = batch["observation.image"]
        S = self.cfg.image_size
        img = F.interpolate(img, size=(S, S), mode="bilinear", align_corners=False)
        img = img * 2 - 1

        with torch.no_grad():
            vis = self.vlm_model.model.vision_model(pixel_values = img).last_hidden_state   # [B, 1024, 768]
            img_tokens = self.vlm_model.model.connector(vis)                               # [B, 64, 960]
            img_tokens = img_tokens * math.sqrt(self.cfg.vlm_hidden)                                     # SmolVLA embed_prefix 스케일

        # 2. Tokenize the text
        text = batch["task"]                     # list[str] 길이 B

        #put the padding on there
        lang = self.tokenizer(text, return_tensors ="pt", padding = "max_length", truncation = True, max_length = self.cfg.max_lang_len)
        input_ids = lang.input_ids.to(img.device)          # tokenizer 출력은 CPU -> 모델 device 로
        lang_pad = lang.attention_mask.to(img.device)
        with torch.no_grad():
            lang_tokens = self.vlm_model.model.text_model.embed_tokens(input_ids)       # [B, 30, 960]
            lang_tokens = lang_tokens * math.sqrt(self.cfg.vlm_hidden)                               # SmolVLA embed_prefix 스케일


        #3. State.
        state_t = batch["observation.state"].float()           # 이미 tensor [B, 2] -> torch.tensor() 다시 감싸면 warning
        state_n = (state_t - self.state_mean) / self.state_std

        #4. State projection.
        state_token = self.state_proj(state_n).unsqueeze(1) # understood

        #5. Prefix.

        prefix = torch.cat([img_tokens, lang_tokens, state_token], dim = 1)


        #6. Padding & Attention mask.
        B = prefix.shape[0]                      # 전역 BATCH_SIZE 대신 실제 batch 크기 사용
        dev = prefix.device
        img_pad = torch.ones(B, img_tokens.shape[1], dtype=torch.bool, device=dev)
        lang_pad_b = lang_pad.bool()
        state_pad = torch.ones(B, state_token.shape[1], dtype=torch.bool, device=dev)

        pad_masks = torch.cat([img_pad, lang_pad_b, state_pad], dim=1)
        
        att_list = ([0] * img_tokens.shape[1] + [0] * lang_tokens.shape[1] + [1] * state_token.shape[1])
        att_masks = torch.tensor(att_list, dtype=torch.bool, device=dev).unsqueeze(0).expand(B, -1)
        
        from utils import make_att_2d_masks
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        
        # 7. Position ids.
        position_ids = torch.cumsum(pad_masks, dim = 1) -1

        return prefix, att_2d_masks, position_ids, pad_masks

    def forward(self, prefix_tokens, att_2d_masks, position_ids):

        # Call the VLM model.
        out = self.vlm_model.model.text_model(
            inputs_embeds=prefix_tokens, # [B, L = 95, D = 960]
            attention_mask=att_2d_masks[:, None, :, :], # [B, 95, 95] -> [B, 1, 95, 95]  4D 여야 HF 가 그대로 사용
            position_ids=position_ids,   # [B, L = 95]
        )

        return out

class ActionExpert(nn.Module):

    def __init__(self, cfg: MyPolicyConfig, action_mean, action_std):
        super().__init__()
        self.cfg = cfg
        D = cfg.expert_hidden
        self.action_in_proj = nn.Linear(cfg.action_dim, D)
        self.action_time_mlp = nn.Sequential(nn.Linear(2 * D, D), nn.SiLU(), nn.Linear(D, D))
        self.action_out_proj = nn.Linear(D, cfg.action_dim)

        self.memory_proj = nn.Linear(cfg.vlm_hidden, D)

        # action normalize 값: 학습(normalize) / 추론(unnormalize) 모두 같은 값 쓰도록 buffer 로 저장
        self.register_buffer("action_mean", torch.as_tensor(action_mean, dtype=torch.float32))
        self.register_buffer("action_std", torch.as_tensor(action_std, dtype=torch.float32))
        
        # Define the decoer layer first and then put 4 om the overall TransformerDecoder.
        self.expert_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=D, 
                                    nhead=cfg.expert_heads, 
                                    dim_feedforward=4 * D,
                                    batch_first=True, 
                                    norm_first=True),
                                    
                                    num_layers=cfg.expert_layers)    
    
    def flow_matching(self, batch, noise=None, t=None):
        '''
        정답 action (x_0) -> 학습 재료 (x_t, t, u_t) 만들기
        action: [B, 16, 2] (normalize 된 정답 chunk)

        x_t = t * noise + (1 - t) * action     # t=0 이면 action, t=1 이면 noise
        u_t = d(x_t)/dt = noise - action       # ground truth 속도 (t 와 무관)
        noise, t 를 넘기면 그걸 사용 (val: 매번 같은 문제로 채점), 없으면 랜덤 (train)
        '''
        action = (batch["action"] - self.action_mean) / self.action_std   # normalize [B, 16, 2]
        B = action.shape[0]
        if noise is None:
            noise = torch.randn_like(action)                                        # [B, 16, 2]
        if t is None:
            t = torch.distributions.Beta(self.cfg.time_beta_alpha, self.cfg.time_beta_beta).sample((B,)).to(action.device) * 0.999 + 0.001   # [B]  (SmolVLA 와 같은 분포)
        tt = t[:, None, None]                                                       # [B, 1, 1] broadcast 용
        x_t = tt * noise + (1 - tt) * action                                        # [B, 16, 2]
        u_t = noise - action                                                        # [B, 16, 2]
        return x_t, t, u_t

    def suffix_preprocessing(self, x_t, t):
        '''
        x_t: [B, 16, 2]  (데이터의 action 이 아니라 noise 섞인 것!)
        t  : [B]         (데이터에 없음, flow_matching 에서 랜덤 샘플)
        '''
        from utils import sinusoidal_time_emb
        a_emb = self.action_in_proj(x_t)                                            # [B, 16, D]
        t_emb = sinusoidal_time_emb(t, a_emb.shape[-1])[:, None, :].expand_as(a_emb)  # [B, 16, D]
        suffix = self.action_time_mlp(torch.cat([a_emb, t_emb], dim = -1))         # [B, 16, D]
        return suffix

    def forward(self, suffix, memory, pad_masks):
        '''
        suffix: [B, 16, D]
        memory: [B, 95, 512]
        pad_masks: [B, 95] should be 1D please.
        '''
        h = self.expert_decoder(tgt = suffix, 
                    memory = memory,
                    memory_key_padding_mask = ~pad_masks)
        v_t = self.action_out_proj(h)
        return v_t


def compute_loss(prefix, expert, batch, cfg, noise=None, t=None):
    prefix_tokens, att_2d_masks, position_ids, pad_masks = prefix.prefix_preprocessing(batch)
    prefix_out = prefix.forward(prefix_tokens, att_2d_masks, position_ids)
    memory = expert.memory_proj(prefix_out.last_hidden_state) # [B, 95, 960] -> [B, 95, 512]  (hidden_states 는 layer별 tuple)
    x_t, t, u_t = expert.flow_matching(batch, noise, t)
    suffix = expert.suffix_preprocessing(x_t, t)
    v_t = expert.forward(suffix, memory, pad_masks)
    valid = (~batch["action_is_pad"])[..., None]                    # [B, 16, 1]
    loss = ((v_t - u_t) ** 2 * valid).sum() / (valid.sum() * cfg.action_dim)
    return loss



@torch.no_grad()
def sample_actions(prefix, expert, batch, cfg):
    '''
    추론: noise (t=1) -> Euler num_steps -> action chunk (t=0)   (Full_script_infer.py Step 3 와 같은 흐름)
    batch: observation.image [B,3,96,96] (0~1), observation.state [B,2], task list[str]
    return: [B, chunk, 2]  unnormalize 된 action (픽셀 좌표)
    '''
    # VLM 은 loop 밖에서 1 번만 (prefix 는 t 와 무관)
    prefix_tokens, att_2d_masks, position_ids, pad_masks = prefix.prefix_preprocessing(batch)
    prefix_out = prefix.forward(prefix_tokens, att_2d_masks, position_ids)
    memory = expert.memory_proj(prefix_out.last_hidden_state)                  # [B, 95, D]

    B = memory.shape[0]
    x_t = torch.randn(B, cfg.chunk_size, cfg.action_dim, device=memory.device)   # t=1 : 순수 noise
    dt = -1.0 / cfg.num_steps
    for step in range(cfg.num_steps):
        t = torch.full((B,), 1.0 + step * dt, device=memory.device)             # 1.0, 0.9, ... 0.1
        suffix = expert.suffix_preprocessing(x_t, t)
        v_t = expert.forward(suffix, memory, pad_masks)                         # [B, 16, 2]
        x_t = x_t + dt * v_t                                                    # Euler

    # unnormalize (학습 때 normalize 한 값 그대로)
    return x_t * expert.action_std + expert.action_mean                         # [B, 16, 2]


def load_policy(ckpt_path, device):
    '''train.py 의 save_ckpt 로 저장한 체크포인트 -> prefix, expert, cfg (eval 모드)'''
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = MyPolicyConfig(**ckpt["cfg"])

    prefix = Prefix(cfg, ckpt["state_mean"], ckpt["state_std"])                 # VLM 은 HF 에서 다시 로드
    prefix.state_proj.load_state_dict(ckpt["state_proj"])

    expert = ActionExpert(cfg, torch.zeros(cfg.action_dim), torch.ones(cfg.action_dim))
    expert.load_state_dict(ckpt["expert"])                                      # action_mean/std buffer 도 여기서 덮어씀

    prefix, expert = prefix.to(device), expert.to(device)
    expert.eval()
    print(f"loaded {ckpt_path} | step {ckpt['step']} | val {ckpt['val_loss']:.4f}")
    return prefix, expert, cfg
