'''
Part2 -> Action Expert ATtention Algorithm
''' 

import torch
from lerobot.configs import PreTrainedConfig
from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_STATE



MODEL_ID = "lerobot/smolvla_base"
DEVICE = "cpu"


#모델 설계도임 이거를 다운받는 것. (레이어 갯수이며, 등등)
config = PreTrainedConfig.from_pretrained(MODEL_ID)
config.device = DEVICE

# create policy model based on that config file.
policy = SmolVLAPolicy.from_pretrained(MODEL_ID, config=config)
policy.eval()

# pre and post processors

pre, post = make_pre_post_processors(policy.config,
                                     pretrained_path = MODEL_ID,
                                     preprocessor_overrides = {"device_processor": {"device": DEVICE}})


####################################
########## Step1 create input ######
####################################

from lerobot.utils.constants import OBS_STATE

raw = {
    "observation.images.camera1": torch.rand(3,256,256),
    "observation.images.camera2": torch.rand(3,256,256),
    "observation.images.camera3": torch.rand(3,256,256),
    OBS_STATE: torch.rand(6), # 6 DOF robot
    "task": "pick up the red cube and place it in the box",
}
batch = pre(raw)

####################################
# Step2 (b) select_action : 한 개씩 + queue
####################################

from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK

images, img_masks = policy.prepare_images(batch)
state = policy.prepare_state(batch)
lang_tokens = batch[OBS_LANGUAGE_TOKENS]
lang_masks  = batch[OBS_LANGUAGE_ATTENTION_MASK]

with torch.no_grad():
    actions_32 = policy.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state)
print("sample_actions out:", actions_32.shape)

actions = actions_32[:, :, :6]
print("unpadded:", actions.shape)

####################################
# Step3 get past_key_values from VLM
####################################
from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK

# batch -> model friendly data.
images, img_masks = policy.prepare_images(batch)
state = policy.prepare_state(batch)
lang_tokens = batch[OBS_LANGUAGE_TOKENS]
lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

# noise plz
bsize = state.shape[0]
model = policy.model # VLA FLOW MATHCIN G소속 model.
noise = model.sample_noise((bsize, model.config.chunk_size, model.config.max_action_dim), state.device)
print("noise:", noise.shape)

# PREPARE PREFIX EMBEDDINGS
prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
    images, img_masks, lang_tokens, lang_masks, state = state)
prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
prefix_position_ids = torch.cumsum(prefix_pad_masks, dim = 1) - 1

# GET KV CACHE FROM THE VLM 
with torch.no_grad():
    _, past_key_values = model.vlm_with_expert.forward(
        attention_mask = prefix_att_2d_masks,
        position_ids = prefix_position_ids,
        past_key_values = None,
        inputs_embeds = [prefix_embs, None],
        use_cache = model.config.use_cache
    )
print("layers:", len(past_key_values))
print("layer0 keys:", past_key_values.layers[0].keys.shape)



###############################################################
########## Next step euler_integrate and denoise_step #########
###############################################################
torch.manual_seed(0)

num_steps = model.config.num_steps # 10 in here
dt = -1.0 / num_steps              # -0.1 (t changes from 1 -> 0)

x_t = noise
with torch.no_grad():
    for step in range(num_steps):
        t = 1.0 + step * dt
        time_tensor = torch.full((bsize,), t , dtype = torch.float32)

        v_t = model.denoise_step(
            prefix_pad_masks = prefix_pad_masks,
            past_key_values = past_key_values,
            x_t = x_t,
            timestep = time_tensor
        )

        #v_t = [B, L, v = 32]
        x_t = x_t + dt * v_t # Euler one step

        print(f"step {step} t={t:.1f} v_t {tuple(v_t.shape)} "
              f"cache_len {past_key_values.get_seq_length()}")


###############################################################
########## Step6 suffix / full attention mask visualization ###
###############################################################
from pathlib import Path
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).parent

# denoise_step 안의 mask 만드는 부분을 그대로 풀어쓰기 (t=1.0 기준, mask는 t와 무관)
with torch.no_grad():
    suffix_embs, suffix_pad_masks, suffix_att_masks = model.embed_suffix(noise, torch.full((bsize,), 1.0))

suffix_len = suffix_pad_masks.shape[1]   # 50
prefix_len = prefix_pad_masks.shape[1]   # 241
prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(bsize, suffix_len, prefix_len)  # [1, 50, 241]
suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)                # [1, 50, 50]
full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)          # [1, 50, 291]

print("suffix_embs  :", suffix_embs.shape)           # [1, 50, 720]
print("prefix_pad_2d:", prefix_pad_2d_masks.shape)
print("suffix_att_2d:", suffix_att_2d_masks.shape)
print("full         :", full_att_2d_masks.shape)

fig, ax = plt.subplots(1, 3, figsize=(18, 4), gridspec_kw={"width_ratios": [prefix_len, suffix_len, prefix_len + suffix_len]})
for a, m, name in zip(ax, [prefix_pad_2d_masks, suffix_att_2d_masks, full_att_2d_masks],
                      ["prefix_pad_2d", "suffix_att_2d", "full"]):
    a.imshow(m[0].float(), cmap="gray", aspect="auto", vmin=0, vmax=1)
    a.set_title(f"{name} {tuple(m.shape)}")
    a.set_xlabel("key")
    a.set_ylabel("query (action token)")
ax[2].axvline(prefix_len - 0.5, color="red")   # KV cache | 새 action token 경계
plt.tight_layout()
plt.savefig(OUT_DIR / "suffix_att_2d_mask.png", dpi=100)
plt.close()
print("saved:", OUT_DIR / "suffix_att_2d_mask.png")


###############################################################
########## Step7 unpad + unnormalize ##########################
###############################################################
x_0 = x_t                               # Euler 10 step 결과 [1, 50, 32] (정규화 공간)
actions_norm = x_0[:, :, :6]            # unpad -> [1, 50, 6] (SO-101 6 DOF)
actions_real = post(actions_norm)       # 역정규화 a * std + mean

print("x_0         :", x_0.shape)
print("unpadded    :", actions_norm.shape)
print("unnormalized:", actions_real.shape)
print("step0 norm  :", actions_norm[0, 0])
print("step0 real  :", actions_real[0, 0])
# smolvla_base는 stats 키가 'so100.buffer.action' 등이라 'action'과 안 맞음 -> identity
print("post() changed values?", not torch.allclose(actions_norm, actions_real))

# action chunk 50 step x 6 DOF 그래프
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
for j in range(6):
    ax[0].plot(actions_norm[0, :, j].numpy(), label=f"dof {j}")
    ax[1].plot(actions_real[0, :, j].numpy(), label=f"dof {j}")
ax[0].set_title("action chunk (normalized)")
ax[1].set_title("action chunk after post() (unnormalized)")
for a in ax:
    a.set_xlabel("chunk step (0~49)")
    a.legend(fontsize=7)
plt.tight_layout()
plt.savefig(OUT_DIR / "action_chunk.png", dpi=100)
plt.close()
print("saved:", OUT_DIR / "action_chunk.png")
