# Step 0 - setting, download pretrained model and config

import torch
from lerobot.configs import PreTrainedConfig
from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

MODEL_ID = "lerobot/smolvla_base"
DEVICE = "cpu"

#모델 설계도임 이거를 다운받는 것. (레이어 갯수이며, 등등)
config = PreTrainedConfig.from_pretrained(MODEL_ID)
config.device = DEVICE

# create policy model based on that config file.
policy = SmolVLAPolicy.from_pretrained(MODEL_ID, config=config)
policy.eval()

# create preprocessor and post processor instance based on the config file.
# 저장된 preprocessor의 device_processor가 "cuda"라서 override로 바꿔 줌
pre, post = make_pre_post_processors(policy.config,
                                     pretrained_path = MODEL_ID,
                                     preprocessor_overrides = {"device_processor": {"device": DEVICE}})


# Step1 create input

from lerobot.utils.constants import OBS_STATE

raw = {
    "observation.images.camera1": torch.rand(3,256,256),
    "observation.images.camera2": torch.rand(3,256,256),
    "observation.images.camera3": torch.rand(3,256,256),
    OBS_STATE: torch.rand(6), # 6 DOF robot
    "task": "pick up the red cube and place it in the box",
}

for k, v in raw.items():
    print(k, v.shape if isinstance(v, torch.Tensor) else v)


# Step2 preprocessor
state_before = raw[OBS_STATE].clone() 

batch = pre(raw)

print("====raw -> batch  ====")
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        print(f"{k}: {v.shape} {v.dtype} {v.device}")
    else:
        print(f"{k}: {v}")


tokens = batch["observation.language.tokens"][0]
mask = batch["observation.language.attention_mask"][0]

print("tokens:", tokens)
print("mask:", mask)


# Step3 Prepare_* 모델구조가 요구하는 형태로 변환 (모델에 맞춤)

images, img_masks = policy.prepare_images(batch)
state = policy.prepare_state(batch)
lang_tokens = batch["observation.language.tokens"]
lang_masks = batch["observation.language.attention_mask"]


print("====batch -> prepare_* ====")

# 이미지: 카메라 3개가 리스트로 묶임
print("images     :", len(images), "x", images[0].shape)
print("img_masks  :", len(img_masks), "x", img_masks[0].shape, img_masks)
print("image range: before", batch["observation.images.camera1"].min().item(), "~", batch["observation.images.camera1"].max().item())
print("             after ", images[0].min().item(), "~", images[0].max().item())

# state: 6 → 32 패딩
print("state      :", batch["observation.state"].shape, "->", state.shape)
print("state[0]   :", state[0])

# 언어: 그대로
print("lang_tokens:", lang_tokens.shape)
print("lang_masks :", lang_masks.shape)

# Step4 Embed_prefix
with torch.no_grad():

    # 이미지 한장이 토큰 몇개가 되는지 따로 확인해보자.
    one_img_tokens = policy.model.vlm_with_expert.embed_image(images[0])
    prefix_embs, prefix_pad_masks, prefix_att_masks = policy.model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks, state
    )

print(" ===== Debugging, step4 Embed_prefix ===== \n")

# print("prefix_embs:\n")
# print(prefix_embs)
# print("prefix_embs   :", prefix_embs.shape, prefix_embs.dtype)
# print("prefix_pad_masks:", prefix_pad_masks.shape)
# print("prefix_att_masks:", prefix_att_masks.shape)

# 241개가 어떻게 구성되는지 확인
n_img = one_img_tokens.shape[1] * len(images)
n_lang = lang_tokens.shape[1]
print(f"prefix 길이 = 이미지 {n_img} + 언어 {n_lang} + state 1 = {n_img + n_lang + 1}")


print("prefix_att_masks:\n")
print(prefix_att_masks)



####Step5 - VLM Inference

from lerobot.policies.common.vla_utils import make_att_2d_masks

# 1D -> 2D mask
prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)

# Check out visulaize tool exists or not. 

# Position IDS??, maybe the position of the padding tokens.
prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1 
'''
토큰       : img  img  pick  up  cube  \n  <pad> <pad>  state
pad_masks  :  1    1    1    1    1    1    0     0      1
cumsum     :  1    2    3    4    5    6    6     6      7
cumsum - 1 :  0    1    2    3    4    5    5     5      6   ← position_ids
'''

with torch.no_grad():
    outputs, past_key_values = policy.model.vlm_with_expert.forward(
        attention_mask = prefix_att_2d_masks,
        position_ids   = prefix_position_ids,
        past_key_values = None,
        inputs_embeds = [prefix_embs, None],
        use_cache = True,
    )

prefix_out, suffix_out = outputs


print("\n ===== Step5 VLM forward ===== ")
print("prefix_att_2d_masks :", prefix_att_2d_masks.shape)
print("prefix_position_ids :", prefix_position_ids.shape)
print("position_ids (언어 끝 ~ state):", prefix_position_ids[0, 200:241].tolist())
print("num_vlm_layers      :", policy.model.vlm_with_expert.num_vlm_layers)
print("prefix_out          :", prefix_out.shape)
print("suffix_out          :", suffix_out)


# Step6 KV cache 확인
print("\n ===== Step6 KV cache ===== ")
print("type          :", type(past_key_values).__name__)
print("num layers    :", len(past_key_values.layers))
print("layer0 keys   :", past_key_values.layers[0].keys.shape)     # [B, kv_heads, L, head_dim]
print("layer0 values :", past_key_values.layers[0].values.shape)
print("layer15 keys  :", past_key_values.layers[15].keys.shape)