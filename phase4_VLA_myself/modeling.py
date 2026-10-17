'''
input: 
- "image" : [3,96,96]
- "text" : "Push the T-shaped block onto the T-shaped target."
- "state" : [222.0, 97.0]

output:
action chunk : [2,]

config is at the end pleasse.

'''
from dataclasses import dataclass
from configuration import Config
import torch
import torch.nn.functional as F
from utills import UtilsForVLA


class SmolVLA_encoder(torch.nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        from transformers import AutoTokenizer, AutoModelForCausalLM, SiglipVisionModel
        self.tokenizer = AutoTokenizer.from_pretrained(config.lm_name)
        self.img_encoder = SiglipVisionModel.from_pretrained(config.vision_name)
        self.lm = AutoModelForCausalLM.from_pretrained(config.lm_name, dtype = torch.float32)
        
        # cut off expert layers. (which is 12)
        self.lm.model.layers = self.lm.model.layers[:self.config.expert_layers]

        # image and state embeddign dimensions -> text embedding dimesion which is lm_hidden
        self.state_projector = torch.nn.Linear(config.state_dim, config.lm_hidden)
        self.mm_projector = torch.nn.Linear(config.vision_hidden, config.lm_hidden)
        self.pad_mask = None
    
    def encode_image(self, img):
        #Note img = [B, 2, 3, 96, 96]
        #lerobot imges are alreay 0~1 rescaled so we don't need to rescale them one more.
        # WE are not using the processor at the beginning because it will be too slow.
        #image = self.img_processor(images = img, do_rescale = False, return_tensors="pt")
        #processing the image my self.. [B,2,3,96,96]
        B,T = img.shape[:2]
        img = img.view(B*T,*img.shape[2:])
        img = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
        img = img * 2 -1 #renormzliae 0~1 -> -1~1
        image_embeds = self.img_encoder(pixel_values = img).last_hidden_state
        image_embeds = self.mm_projector(image_embeds) # [2 * B, 196, hidden]
        image_embeds = image_embeds.view(B,-1,image_embeds.shape[-1]) # [B, 2*196, hidden]
        return image_embeds
    
    def emb_prefix(self, image, text, state):
        '''
        it's all about the paddgins 
        I remeber vividly paddings goes eto the stat eand the text propt right? 
        '''

        image_embeds = self.encode_image(image)
        B = image_embeds.shape[0]

        text_tok = self.tokenizer(text, 
                                return_tensors="pt", 
                                max_length=self.config.max_lang_len, 
                                padding="max_length",
                                truncation=True).to(image_embeds.device)
        text_embeds = self.lm.get_input_embeddings()(text_tok.input_ids)
        state_embeds = self.state_projector(state.view(B, 1, -1)) # or unsqueeze(1)

        prefix = torch.cat([image_embeds, text_embeds, state_embeds], dim=1)
            
        #image_embeds의 토큰 길이 만큼 앞을 1로 채우고, 그리고 이제 staet 토큰 길이만큼 1을 뒤로 채운다.
        # B = image_embeds.shape[0]     # 위 (89줄) 에서 이미 구함
        padding_mask = torch.cat([
            text_tok.attention_mask.new_ones(B, image_embeds.shape[1]),  # 이미지: 전부 1
            text_tok.attention_mask,                                     # 텍스트: 실제 토큰 1, pad 0
            text_tok.attention_mask.new_ones(B, state_embeds.shape[1]),  # state: 1
        ], dim=1)

        #Store the padding mask for the suffix.
        self.pad_mask = padding_mask.bool()
        # att_1d_mask = torch.cat([
        #     torch.zeros(B, image_embeds.shape[1], dtype=torch.bool),  # 이미지: 전부 0
        #     torch.zeros(B, text_tok.attention_mask.shape[1], dtype=torch.bool),                                     # 텍스트: 실제 토큰 1, pad 0
        #     torch.ones(B, state_embeds.shape[1], dtype=torch.bool),  # state: 1
        # ], dim=1)
        # 1 = 새 블록 시작 (앞 블록은 이 블록을 못 봄), 0 = 앞 토큰과 같은 블록
        dev = image_embeds.device
        self.att_1d_mask = torch.cat([
            torch.zeros(B, image_embeds.shape[1], dtype=torch.bool, device=dev),             # 이미지: 0 (img + lang 한 블록)
            torch.zeros(B, text_tok.attention_mask.shape[1], dtype=torch.bool, device=dev),  # 텍스트: 0 (pad 여부는 padding_mask 담당)
            torch.ones(B, state_embeds.shape[1], dtype=torch.bool, device=dev),              # state: 1 (새 블록 -> img/lang 은 state 못 봄)
        ], dim=1)

        #JUst MAke 2D mask form the 1D padding adn the attention mask.
        # att_2d_masks = self.make_att_2d_masks(padding_mask, att_1d_mask)
        att_2d_masks = UtilsForVLA.make_att_2d_masks(self.pad_mask, self.att_1d_mask)   # long(0/1) -> bool, 결과도 bool [B, L, L]

        # HF 에 넘길 4D mask: [B, L, L] -> [B, 1, L, L] (head 축), bool -> float (볼 수 있음 0 / 못 봄 -inf)
        # 4D mask 는 HF 가 손 안 대고 그대로 씀 (masking_utils: "already 4D ... returned as-is")
        att_2d_masks = att_2d_masks.unsqueeze(1)
        mask_4d = torch.where(att_2d_masks, 0.0, torch.finfo(prefix.dtype).min)
        #torch.where(cond, x, y)     # cond 가 True 인 칸은 x, False 인 칸은 y

        #position ids (pad 자리는 번호 건너뜀 -> state 가 text 바로 다음 번호)
        position_ids = padding_mask.cumsum(dim = -1) - 1

        # return prefix, padding_mask, position_ids, att_2d_masks
        # 다음 단계: self.lm.model(inputs_embeds=prefix, attention_mask=mask_4d, position_ids=position_ids, use_cache=True)
        return prefix, position_ids, mask_4d


    def forward(self, image, text, state):
        prefix, position_ids, mask_4d = self.emb_prefix(image, text, state)
        return self.lm.model(inputs_embeds=prefix, attention_mask=mask_4d, position_ids=position_ids, use_cache=True)


class SmolVLA_expert(torch.nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # self.action_embedding = torch.nn.Linear(config.max_action_dim, config.expert_hidden)
        # self.time_embedding = torch.nn.Linear(config.num_timesteps, config.expert_hidden)
        self.action_embedding = torch.nn.Linear(config.action_dim, config.expert_hidden)   # 2 -> 384
        # time 은 sinusoidal_time_emb (학습 파라미터 없음) -> Linear 필요 없음
        # action + time 섞기 (SmolVLA action_time_mlp_in/out 과 같음): [B,16,768] -> [B,16,384]
        self.action_time_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * config.expert_hidden, config.expert_hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(config.expert_hidden, config.expert_hidden),
        )
        from transformer import ExpertTransformer
        self.expert_transformer = ExpertTransformer(config)

    def embeddings(self, x_t, t):

        # [x_t, t] , x_t and t is [action_len, d_expert_hidden] each. -> so we need MLP
        action_emb = self.action_embedding(x_t)                                      # [B, 16, 384]
        time_emb = UtilsForVLA.sinusoidal_time_emb(t, self.config.expert_hidden)     # [B, 384]
        time_emb = time_emb[:, None, :]                                              # [B, 1, 384]
        time_emb = time_emb.expand_as(action_emb)                                    # [B, 16, 384] (16 토큰에 같은 t)
        
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)                  # [B, 16, 768]
        suffix = self.action_time_mlp(action_time_emb)                             # [B, 16, 384]
        
        return suffix

    def forward(self, x_t, t, KV_cache, prefix_pad_mask, pefix_att_1D_mask):

        suffix = self.embeddings(x_t, t) #[B, 16, 384]
        suffix_pad_mask = torch.ones(B, suffix.shape[1], dtype = torch.bool, device = x_t.device) # [B, 16] should be like that.
        suffix_pad_mask = torch.cat([prefix_pad_mask, suffix_pad_mask], dim = 1)

        # ATtention mask please in here. multitransofrmer block needs,
        # 1. suffix_emb, 2. 2D attent mask, 3. padding mask. -> create att_2d_mask in here.
        B = suffix.shape[0]
        suffix_att_mask = torch.ones(B, suffix.shape[1], dtype = torch.bool, device = x_t.device) # [B, 16] should be like that.
        suffix_att_mask = torch.cat([pefix_att_1D_mask, suffix_att_mask], dim = 1)
        att_2d_mask = UtilsForVLA.make_att_2d_masks(suffix_pad_mask, suffix_att_mask)
        out = self.expert_transformer(suffix, att_2d_mask, KV_cache)

        return


from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ROOT = Path(__file__).resolve().parents[1]          # VLA_study_myself/
DATA_DIR = ROOT / "data" / "pusht"                  # 이미 받아둔 로컬 데이터 (다운로드 X)

def main_infer():
    config = Config()

    #Step1 just get a single data from the dataset.
    ds = LeRobotDataset("lerobot/pusht", root=DATA_DIR)
    frame0, frame1 = ds[0], ds[1]
    image = torch.stack([frame0["observation.image"], frame1["observation.image"]], dim=0).unsqueeze(0)  # [1, 2, 3, 96, 96]
    text = frame0["task"]
    state = frame0["observation.state"]

    #Step2 encode the image, text, state.
    model = SmolVLA_encoder(config)
    out = model(image, text, state)

    # We need Cache and pad mask for the suffix.
    cache = out.past_key_values
    print("# of layers", len(cache))
    print("K", cache.layers[0].keys.shape)
    print("V", cache.layers[0].values.shape) # [1,3,413,64]
    print("V", cache.layers[0].values.shape) # [1,3,413,64]


    #Step3 Expert — inference starts at pure noise, t=1 (Euler then walks t toward 0)
    B = image.shape[0]
    x_t = torch.randn(B, config.chunk_size, config.action_dim)  # [B, 16, 2]
    t = torch.ones(B)                                           # [B], t=1

    expert = SmolVLA_expert(config)

    expert_step = 10
    T = 1 
    dt = T / expert_step

    for i in range(expert_step):
        v_t = expert(x_t, t, cache, model.pad_mask, model.att_1d_mask) # we got velocity from the exerpt.
        x_t = x_t + v_t * dt # update the state.

    #Final action chunck is x_t


    return



if __name__ == "__main__":
    main_infer()