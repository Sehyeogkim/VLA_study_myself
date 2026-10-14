from pathlib import Path
import pandas as pd
import numpy as np
import torch

# Step 0. Load PushT
# parquet lives at data/pusht/data/chunk-000/file-000.parquet
# images are not in the parquet; they are the mp4 next to it
ROOT = Path(__file__).resolve().parents[1]          # VLA_study_myself/
DATA_DIR = ROOT / "data" / "pusht"
PARQUET = DATA_DIR / "data" / "chunk-000" / "file-000.parquet"

data = pd.read_parquet(PARQUET)
print("rows, cols:", data.shape)
print("columns:", list(data.columns))
print(data.dtypes)
print(data.head())

# Video -> 1 frame 디멘션 확인하기 -> [96,96,3]
# import av

# VIDEO = DATA_DIR / "videos" / "observation.image" / "chunk-000" / "file-000.mp4"
# with av.open(VIDEO) as container:
#     frame = next(container.decode(video=0))
# image = frame.to_ndarray(format="rgb24")  # (H, W, C)
# print("video:", VIDEO.name)
# print("frame0:", image.shape, image.dtype)


# Step 1. Load the image
# jus tmake a fake a image as same as the data (video를 우리가 1frame을 추출했을때 사진 디멘션을 그대로 input으로 일단 구조부터 짜자.)
# 그리고 이제 state [s1,s2] 인거 확인했고 Action = [a1,a2] 인거 확인했고 
# task = "push" 이걸로 그냥 하자. 고정 하자. task = push 이걸로 ㅇㅋ?

########################################################
########################################################
import torch
state = np.asarray(data["observation.state"].iloc[0])
action = np.asarray(data["action"].iloc[0])
task = "Push the T-shaped block onto the T_shaped target"

# preprocessing and change to the embedding vector
# embedding vector dimesion -> 960

'''
SigVLM 인가 그거 input dimension을 맞출라고 강제로 밑에 작업을 진행함.
사용하는 VLM model에 맞게 input dimension을 맞춰줘야 함.
'''
# [H,W,C] -> [C,H,W] -> [1,C,H,W]
import torch.nn.functional as F
image = torch.rand((96,96,3))
img = image.permute(2,0,1).unsqueeze(0)
img = F.interpolate(img, size = (512,512), mode = "bilinear", align_corners = False)
img = (img * 2) - 1
print("img after the range adjustment:", img.shape)

# WE need to put image -> image encoder -> img_tokens.
from transformers import AutoModelForImageTextToText
vlm_model = AutoModelForImageTextToText.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
vlm_model.eval()


'''
알아두면 유용한 것, vlm_model -> model -> [vision_model, text_model, connector]
'''

# image -> encoder -> connector -> img_tokens
with torch.no_grad():
    # encoder 에서 갓 나옴 (encoder 도 트렌스포머이다.)
    vis = vlm_model.model.vision_model(pixel_values = img).last_hidden_state
    # encoder output -> connector -> img_tokens
    img_tokens = vlm_model.model.connector(vis)
print("vision_model out", vis.shape)
print("img_tokens shape:", img_tokens.shape)

# text -> text_tokens
'''
text -> (tokenizer) -> input_ids -> (embed_tokens, this is dictonary) -> lang_tokens
'''
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
#input_ids = tokenizer(task, return_tensors = "pt").input_ids # this is the basic version

# LEt us try to put the padding on there
lang = tokenizer(task, return_tensors ="pt", padding = "max_length", truncation = True, max_length = 30)

with torch.no_grad():
    input_ids = lang.input_ids
    lang_pad = lang.attention_mask
    lang_tokens = vlm_model.model.text_model.embed_tokens(input_ids)
    print("lang_tokens shape:", lang_tokens.shape)
    print("input_ids value:", input_ids)
    print("attention_mask value:", lang_pad)

# now states
import json

stats_json = json.load(open(DATA_DIR / "meta" / "stats.json"))
mean = torch.tensor(stats_json["observation.state"]["mean"], dtype=torch.float32)
std = torch.tensor(stats_json["observation.state"]["std"], dtype=torch.float32)


# CHange to the tensor and unsqueeze to the batch dimension and normalize the state.
state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
state_n = (state_t - mean) / std 
print("state_n shape:", state_n.shape)
print("===============================================\n\n")

import torch.nn as nn
HIDDEN = 960
state_proj = nn.Linear(2, HIDDEN)
state_token = state_proj(state_n).unsqueeze(1) # understood

# now prefix = [img_tokens, lang_tokens, state_token]
# printout all the dimesino so far
print("img_tokens shape:", img_tokens.shape)
print("lang_tokens shape:", lang_tokens.shape)
print("state_token shape:", state_token.shape)

# now prefix = [img_tokens, lang_tokens, state_token]
'''
let us put the padding on the text token.
'''
prefix = torch.cat([img_tokens, lang_tokens, state_token], dim = 1)
print("prefix shape:", prefix.shape)


########################################################
########################################################

# 두번째 스탭으론 넘어가자. 이제 실제로 이제 다음 process는 우리가 이제 KV CAHCE 제작이다.
# 그러니까 이제 저 prefix -> (text_model which is a Transformer) -> KV Cache

'''
1. pad_masks
2. att_masks
3. position_ids
4. text_model output -> KV Cache
'''

from utils import make_att_2d_masks, save_mask_png

B = prefix.shape[0]

# Pad Mask : 전부 bool, 전부 [B, n]
img_pad = torch.ones(B, img_tokens.shape[1], dtype=torch.bool)      # [B, 64]
lang_pad_b = lang_pad.bool()                                        # [B, 30] tokenizer 가 이미 batch 포함
state_pad = torch.ones(B, state_token.shape[1], dtype=torch.bool)   # [B, 1]

pad_masks = torch.cat([img_pad, lang_pad_b, state_pad], dim=1)      # [B, 95]

# Attention Mask : list -> tensor [1, L] -> batch 만큼 expand [B, L]
att_list = ([0] * img_tokens.shape[1] + [0] * lang_tokens.shape[1] + [1] * state_token.shape[1])
att_masks = torch.tensor(att_list, dtype=torch.bool).unsqueeze(0).expand(B, -1)   # [B, 95]

att_2d_masks = make_att_2d_masks(pad_masks, att_masks)              # [B, 95, 95]
print("pad_masks shape:", pad_masks.shape)
print("att_masks shape:", att_masks.shape)
print("att_2d_masks shape:", att_2d_masks.shape)


# # visulzie the mask please
# OUT_DIR = Path(__file__).parent / "results"
# OUT_DIR.mkdir(exist_ok=True)
# n_img, n_lang = img_tokens.shape[1], lang_tokens.shape[1]
# save_mask_png(att_2d_masks[0], OUT_DIR / "prefix_att_2d_mask.png",
#               boundaries=[n_img, n_img + n_lang],
#               title="prefix [img | lang | state]")



########################################################
# Position ids (Understood)
position_ids = torch.cumsum(pad_masks, dim = 1) -1

# prefix -> text_model (LlamaModel, 32 layer) -> KV cache
# transformers/models/llama/modeling_llama.py 의 LlamaModel
# attention_mask 를 4D [B, 1, L, L] 로 주면 HF 가 causal mask 를 안 만들고 그대로 사용
with torch.no_grad():
    out = vlm_model.model.text_model(
        inputs_embeds=prefix,                          # [B, 95, 960]
        attention_mask=att_2d_masks[:, None, :, :],    # [B, 1, 95, 95]  (1 = head 축, 모든 head 공유)
        position_ids=position_ids,                     # [B, 95]
        use_cache=True,
    )

# out has 'last hiddenstate', 'past_key_values', 'hidden_states', 'attentions'
past_key_values = out.past_key_values
print("last_hidden_state:", out.last_hidden_state.shape)          # [B, 95, 960]
print("KV cache layers  :", len(past_key_values.layers))           # 32
print("layer0 keys      :", past_key_values.layers[0].keys.shape)  # [B, kv_heads, 95, head_dim]
print("layer0 values    :", past_key_values.layers[0].values.shape)


### B only got the output of the VLM model (text_model)
vlm_out = out.last_hidden_state
print("vlm_out shape:", vlm_out.shape)


########################################################
#### Now we are going to the action expert model #######
########################################################
'''
get x_t
for step in range(T)
    t = 1 + step * dt
    suffix_emb = embedding(x_t) + embedding(t)
    input = [suffix_emb, vlm_out] ?? self attention

    h = expert(input)
    v_t = action_out_proj(h)
    x_t = x_t + dt * v_t

actions = unnormalize(x)
'''

########################################################
# Step 3. Action Expert (B: cross-attention)  — 추론 흐름
########################################################

'''
A: Cross attention with KV past Cache
B: Cross - attention with the hidden state of the VLM Model
C: Joint attention?? will be dealt later on.
'''

CHUNK = 16 # Num of Action chunk.
ACTION_DIM = 2
D = 512  # Expert hidden (less than VLM)
T = 10   # Denois step number


# 전부 학습 대상
mem_proj = nn.Linear(960, D)
memory = mem_proj(vlm_out) # 960 -> 512 change the dimension for the expert

x_t = torch.randn(B, CHUNK, ACTION_DIM)
dt = -1.0 / T # 1 -> 0

with torch.no_grad():
    for step in range(T):
        t = torch.full((B,), 1.0 + step * dt)
        print("t shape:", t.shape)
        print("t value:", t)

        # suffix = concat + MLP
        action_in_proj = nn.Linear(ACTION_DIM, D)
        a_emb = action_in_proj(x_t) # [B, CHUNK, D]
        from utils import sinusoidal_time_emb

        # t = [B] -> [B,D] -> [B, CHUNK, D]
        t_emb = sinusoidal_time_emb(t, D)[:, None, :].expand(-1, CHUNK, -1) # [B, CHUNK, D]
        

        # we Saw this on the SmolVLA paper.
        action_time_mlp = nn.Sequential(nn.Linear(2 * D, D), nn.SiLU(), nn.Linear(D, D))
        suffix = action_time_mlp(torch.cat([a_emb, t_emb], dim = -1))
        print("suffix shape:", suffix.shape)

        # Now put the suffix and the memory into th model

        expert = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=D, 
                                    nhead=8, 
                                    dim_feedforward=4 * D,
                                    batch_first=True, 
                                    norm_first=True),
                                    num_layers=4)
        
        # "Attention is all you need" -> 원조 transformer decoder 구조
        # And in here we are using the cross attention only
        # memory -> k, v inside the transformer decoder layer
        h = expert(tgt = suffix, 
                    memory = memory,
                    memory_key_padding_mask = ~pad_masks)
        print("h shape:", h.shape)
        
        # h -> v_t
        action_out_proj = nn.Linear(D, ACTION_DIM)
        v_t = action_out_proj(h)
        print("v_t shape:", v_t.shape)

        # v_t -> x_t
        x_t = x_t + dt * v_t
        print("x_t shape:", x_t.shape)

#### 이제 training loop 으로 들어가보자. 여기서 지금까지는 추론 흐름이었다.
#### 이제 training loop 으로 들어가보자. 여기서 지금까지는 추론 흐름이었다.