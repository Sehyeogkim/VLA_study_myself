from dataclasses import dataclass
from configuration import Config
import torch
import torch.nn.functional as F
from utills import UtilsForVLA

'''
 Groot 1.7 build my self.
'''

class Groot_VLM(torch.nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration, Qwen3VLProcessor
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        self.tokenizer.padding_side = config.padding_side
        self.processor = Qwen3VLProcessor.from_pretrained(config.backbone_name)
        self.processor.tokenizer = self.tokenizer
        self.vlm = Qwen3VLForConditionalGeneration.from_pretrained(
            config.backbone_name, dtype=getattr(torch, config.dtype)
        )
        self.pad_mask = None
        self.att_1d_mask = None

    def prepare_images(self, img, train=False):
        '''
            img: [B, 2, 3, 256, 256] float 0~1. 2 = 3인칭, 손목.
            230 창을 다시 256 으로 늘린다. 색은 여기서 안 건드린다.
            train=True  : 샘플마다 창 위치를 뽑고, 카메라 두 대는 그 위치를 같이 씀.
            train=False : 가운데. 평가.
            uint8 로 넘기면 Qwen 프로세서가 /255 와 CLIP mean/std 를 한다.
        '''
        B, n_cam, C, _, _ = img.shape
        flat = img.reshape(B * n_cam, C, img.shape[-2], img.shape[-1])
        target = self.config.image_size
        crop = self.config.image_crop_size
        if flat.shape[-2] != target or flat.shape[-1] != target:
            flat = F.interpolate(flat, size=(target, target), mode="bicubic", align_corners=False)
        views = flat.view(B, n_cam, C, target, target)
        slack = target - crop
        if train:
            tops = torch.randint(0, slack + 1, (B,))
            lefts = torch.randint(0, slack + 1, (B,))
        else:
            tops = torch.full((B,), slack // 2)
            lefts = torch.full((B,), slack // 2)
        windows = [
            views[b, :, :, int(tops[b]):int(tops[b]) + crop, int(lefts[b]):int(lefts[b]) + crop]
            for b in range(B)
        ]
        flat = torch.stack(windows).reshape(B * n_cam, C, crop, crop)
        flat = F.interpolate(flat, size=(target, target), mode="bicubic", align_corners=False).clamp(0, 1)
        return (flat * 255).round().to(torch.uint8).view(B, n_cam, C, target, target)

    def build_conversation(self, text, n_cam):
        '''
        카메라 n_cam 장 + 문장 하나 -> Qwen 채팅 문자열.
        text: str 하나, 또는 배치 길이만큼의 list[str].
        '''
        if isinstance(text, str):
            text = [text]
        prompts = []
        for sentence in text:
            conversation = [{
                "role": "user",
                "content": [
                    *[{"type": "image"} for _ in range(n_cam)],
                    {"type": "text", "text": sentence},
                ],
            }]
            prompts.append(self.processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=False,
            ))
        return prompts

    def forward(self, image, text, train=False):
        '''
        state 는 VLM 에 안 넣는다. 사진 두 장 + 문장만.
        반환 last_hidden_state 는 select_layer 번째 블록 hidden.
        '''
        uint8 = self.prepare_images(image, train=train)
        B, n_cam = uint8.shape[:2]
        frames = [
            uint8[b, v].permute(1, 2, 0).cpu().numpy()
            for b in range(B)
            for v in range(n_cam)
        ]
        texts = self.build_conversation(text, n_cam)
        encoded = self.processor(text=texts, images=frames, padding=True, return_tensors="pt")
        encoded = {k: v.to(self.vlm.device) for k, v in encoded.items() if torch.is_tensor(v)}
        out = self.vlm(**encoded, output_hidden_states=True, return_dict=True)
        hidden = out.hidden_states[self.config.select_layer]
        self.pad_mask = encoded["attention_mask"].bool()
        # N1.7 DiT 는 이 토큰 전체에 cross-attn. 예전 블록 마스크는 전부 한 블록.
        self.att_1d_mask = torch.zeros_like(self.pad_mask)
        out.last_hidden_state = hidden
        return out


class _AdaLN(torch.nn.Module):
    '''
        Adaptive Layer Normalization.
        같은 네트워크가 t에 따라 다른 속도를 내게 한다.
        x * (1 + scale) + shift , time -> scale, shift.
        
    '''
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.norm = torch.nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = torch.nn.Linear(cond_dim, dim * 2)

    def forward(self, x, cond):
        scale, shift = self.proj(cond).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]


class _DiTBlock(torch.nn.Module):
    '''self-attn(state+action) 다음에 VLM hidden 으로 cross-attn. 시간은 AdaLN.'''

    def __init__(self, dim, heads, cross_dim, ffn_dim):
        super().__init__()
        self.adaln_self = _AdaLN(dim, dim)
        self.adaln_cross = _AdaLN(dim, dim)
        self.adaln_ff = _AdaLN(dim, dim)
        self.self_attn = torch.nn.MultiheadAttention(dim, heads, batch_first=True)
        self.cross_attn = torch.nn.MultiheadAttention(
            dim, heads, kdim=cross_dim, vdim=cross_dim, batch_first=True
        )
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(dim, ffn_dim),
            torch.nn.GELU(),
            torch.nn.Linear(ffn_dim, dim),
        )

    def forward(self, x, context, cond, context_pad):
        '''
            context - Normed hidden state from VLM.
            cond    - Denoising time condition.
        '''
        h = self.adaln_self(x, cond)
        h = self.self_attn(h, h, h, need_weights=False)[0] + x
        y = self.adaln_cross(h, cond)
        y = self.cross_attn(
            y, context, context, key_padding_mask=~context_pad, need_weights=False
        )[0] + h
        return self.ff(self.adaln_ff(y, cond)) + y


class Groot_DiT(torch.nn.Module):
    '''
    N1.7 action head.
    state 1토큰 + noisy action 을 붙이고, VLM hidden 을 cross-attn 으로 읽는다.
    출력은 action 칸의 속도 v_t. [B, horizon, action_dim]
    '''

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        dim = config.dit_heads * config.dit_head_dim
        self.dim = dim
        self.vlm_norm = torch.nn.LayerNorm(config.backbone_hidden)
        self.state_encoder = torch.nn.Sequential(
            torch.nn.Linear(config.state_dim, dim),
            torch.nn.SiLU(),
            torch.nn.Linear(dim, dim),
        )
        self.action_encoder = torch.nn.Linear(config.action_dim, dim)
        self.action_pos = torch.nn.Embedding(config.action_horizon, dim)
        self.blocks = torch.nn.ModuleList([
            _DiTBlock(dim, config.dit_heads, config.backbone_hidden, dim * 4)
            for _ in range(config.dit_layers)
        ])
        self.action_decoder = torch.nn.Sequential(
            torch.nn.Linear(dim, config.dit_hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(config.dit_hidden, config.action_dim),
        )

    def _time_cond(self, t):
        return UtilsForVLA.sinusoidal_time_emb(t, self.dim)

    def forward(self, state, x_t, t, vlm_hidden, pad_mask, att_1d_mask=None):
        if state.ndim == 3:
            state = state[:, -1]
        
        # state -> token. [B, 1, dim]
        state_tok = self.state_encoder(state)[:, None, :]
        
        # action -> token. [B, 16, dim]
        action_tok = self.action_encoder(x_t)
        
        # action token에 position encoding 을 더한다. [B, 16, dim]
        pos = torch.arange(action_tok.shape[1], device=x_t.device)
        action_tok = action_tok + self.action_pos(pos)[None]
        
        # state와 action token을 연결한다. [B, 17, dim]
        tokens = torch.cat([state_tok, action_tok], dim=1)
        
        # Denoising time condition 을 계산한다. [B, dim]
        cond = self._time_cond(t)
        
        # VLM hidden 을 정규화한다. [B, seq_len, dim]
        context = self.vlm_norm(vlm_hidden)
        
        # pad mask 를 bool 로 변환한다.
        if pad_mask.dtype != torch.bool:
            pad_mask = pad_mask.bool()
        for block in self.blocks:
            tokens = block(tokens, context, cond, pad_mask)
        
        return self.action_decoder(tokens[:, 1:])


class GrootPolicy(torch.nn.Module):
    '''
    VLM 한 번 + DiT flow matching.
    학습 forward 는 masked MSE. sample_actions 는 t=1 noise 에서 Euler 로 action chunk.
    '''

    def __init__(self, config: Config, stats: dict):
        super().__init__()
        self.config = config
        self.backbone = Groot_VLM(config)
        self.dit = Groot_DiT(config)
        if not config.train_backbone:
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        f32 = lambda x: torch.tensor(x, dtype=torch.float32)
        self.register_buffer("state_mean", f32(stats["observation.state"]["mean"]))
        self.register_buffer("state_std", f32(stats["observation.state"]["std"]).clamp(min=1e-6))
        self.register_buffer("action_mean", f32(stats["action"]["mean"]))
        self.register_buffer("action_std", f32(stats["action"]["std"]).clamp(min=1e-6))

    def train(self, mode=True):
        super().train(mode)
        if not self.config.train_backbone:
            self.backbone.eval()
        return self

    @staticmethod
    def stack_cameras(image, image2):
        '''카메라 두 대를 시간 축이 아니라 dim=1 에 붙인다. [B, 2, 3, H, W]'''
        if image.ndim == 5:
            image = image[:, -1]
            image2 = image2[:, -1]
        return torch.stack([image, image2], dim=1)

    @staticmethod
    def current_state(state):
        if state.ndim == 3:
            state = state[:, -1]
        return state

    def encode(self, image, text, train):
        if self.config.train_backbone:
            out = self.backbone(image, text, train=train)
        else:
            with torch.no_grad():
                out = self.backbone(image, text, train=train)
        hidden = out.last_hidden_state
        return hidden, self.backbone.pad_mask

    def velocity(self, state, x_t, t, hidden, pad_mask):
        hidden = hidden.to(device=x_t.device, dtype=torch.float32)
        pad_mask = pad_mask.to(device=x_t.device)
        return self.dit(state, x_t, t, hidden, pad_mask)

    def flow_targets(self, action):
        noise = torch.randn_like(action)
        t = UtilsForVLA.sample_time_beta(action.shape[0], action.device)
        tt = t[:, None, None]
        x_t = tt * noise + (1 - tt) * action
        u_t = noise - action
        return x_t, t, u_t

    def masked_mse(self, v_t, u_t, action_is_pad):
        valid = (~action_is_pad)[..., None]
        return ((v_t - u_t) ** 2 * valid).sum() / (valid.sum() * self.config.action_dim)

    def forward(self, batch):
        image = self.stack_cameras(batch["observation.images.image"], batch["observation.images.image2"])
        state = self.current_state(batch["observation.state"])
        action = batch["action"]
        state = (state - self.state_mean) / self.state_std
        action = (action - self.action_mean) / self.action_std
        if self.training and self.config.state_dropout > 0:
            drop = torch.rand(state.shape[0], device=state.device) < self.config.state_dropout
            state = state.masked_fill(drop[:, None], 0.0)

        hidden, pad_mask = self.encode(image, batch["task"], train=self.training)
        x_t, t, u_t = self.flow_targets(action)
        v_t = self.velocity(state, x_t, t, hidden, pad_mask)
        return self.masked_mse(v_t, u_t, batch["action_is_pad"])

    @torch.no_grad()
    def sample_actions(self, image, text, state):
        '''
        image: [B, 2, 3, 256, 256] 0~1
        text : list[str]
        state: [B, 8]  정규화 전
        return [B, horizon, 7]  데이터와 같은 스케일의 action
        '''
        state = (state - self.state_mean) / self.state_std
        hidden, pad_mask = self.encode(image, text, train=False)
        B = state.shape[0]
        x_t = torch.randn(B, self.config.chunk_size, self.config.action_dim, device=state.device)
        t = torch.ones(B, device=state.device)
        dt = -self.config.T / self.config.num_denoise_steps
        for _ in range(self.config.num_denoise_steps):
            v_t = self.velocity(state, x_t, t, hidden, pad_mask)
            x_t = x_t + dt * v_t
            t = t + dt
        return x_t * self.action_std + self.action_mean


from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ROOT = Path(__file__).resolve().parents[1]          # VLA_study_myself/
# DATA_DIR = ROOT / "data" / "pusht"                  # 이미 받아둔 로컬 데이터 (다운로드 X)
DATA_DIR = ROOT / "data" / "libero"

def main_infer():
    
    config = Config()

    #Step1 just get a single data from the dataset.
    ds = LeRobotDataset(
        "lerobot/libero",
        root=DATA_DIR,
        episodes=[0],
        download_videos=False,
        video_backend="pyav",
    )

    frame = ds[0]
    image = torch.stack(
        [frame["observation.images.image"], frame["observation.images.image2"]],
        dim=0,
    ).unsqueeze(0)                                          # [1, 2, 3, 256, 256]
    # text = frame["task"].unsqueeze(0)
    text = [frame["task"]]
    state = frame["observation.state"].unsqueeze(0)        # [1, 8]

    #Step2 encode the image, text, state.
    model = Groot_VLM(config)
    out = model(image, text)

    # We need Cache and pad mask for the suffix.
    vlm_hidden = out.last_hidden_state
    print("VLM hidden state shape:", vlm_hidden.shape)

    #Step3 Expert — inference starts at pure noise, t=1 (Euler then walks t toward 0)
    B = image.shape[0]
    x_t = torch.randn(B, config.chunk_size, config.action_dim)  # [B, 16, 2]
    t = torch.ones(B)                                           # [B], t=1

    expert = Groot_DiT(config)

    dt = -config.T / config.num_denoise_steps

    for i in range(config.num_denoise_steps):
        v_t = expert(state, x_t, t, vlm_hidden, model.pad_mask, model.att_1d_mask) # we got velocity from the exerpt.
        x_t = x_t + v_t * dt # update the state.
        t = t + dt # update the time.

    #Final action chunck is x_t
    return x_t



if __name__ == "__main__":
    main_infer()