'''
04 Training SmolVLA - 학습 1 step 을 손으로 풀어보기

[전체 흐름]
  dataset 1 frame = (이미지 2장 + state + 문장) -> 정답 action chunk [50, 6]
       │  DataLoader (B 개 랜덤 frame)  +  pre() (rename / 정규화 / 토크나이즈)
       ▼
  Step1  batch           action [B,50,6], action_is_pad [B,50], image, state, tokens
  Step2  flow matching   t ~ Beta(1.5,1), noise ~ N(0,1)
                         x_t = t*noise + (1-t)*a   (모델 입력)
                         u_t = noise - a           (정답 velocity = dx_t/dt)
  Step3  loss            [prefix, suffix] 한 번에 forward (KV cache 없음) -> v_t
                         mse(u_t, v_t) -> 6차원만 -> pad 마스크 -> 유효칸 평균
  Step4  backward        .grad 는 학습 대상(약 100M, 22%)에만 생김. VLM 은 freeze
  Step5  optimizer       zero_grad -> forward -> backward -> clip(10) -> step  (x20, 같은 batch overfit)
  Step6  lr schedule     warmup 1k (1e-7 -> 1e-4) + cosine (-> 2.5e-6)  ※ 그래프만 (학습 X, 실제 적용은 05)

[실행]
  로컬(CPU) : Step5 가 매우 느림 (450M forward+backward)
      HF_HUB_OFFLINE=1 uv run --project /Users/jeff/project/lerobot python phase1_smolvla/04_training.py
  RunPod(GPU): DEVICE 가 자동으로 cuda 선택됨. 결과 -> results/

[RunPod A40 결과]  (results/04_training_runpod_A40.log, results/04_train_loss_lr.png)
  Step3  my loss == policy.forward loss  (일치 True)
  Step4  학습 99,880,992 / 450,046,176 (22.2%)
         VLM(vision/connector/text/lm_head) grad 0, expert + proj + state_proj grad O
  Step5  loss 0.87 -> 0.04 (20 step). step1 에서 3.35 로 튐 = warmup 없이 Adam 첫 step 이 전 파라미터를 lr 만큼 이동
  Step6  lr @0=1e-7, @1000=1e-4, @15000=5.1e-5, @30000=2.5e-6
'''
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from lerobot.configs import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.common.flow_matching import FlowConvention, make_flow_matching_inputs
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.processor.rename_processor import rename_batch_keys, rename_stats
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

ROOT = Path(__file__).resolve().parents[1]          # VLA_study_myself/
DATA_DIR = ROOT / "data" / "svla_so101_pickplace"
OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(exist_ok=True)

MODEL_ID = "lerobot/smolvla_base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
B = 2

# 데이터 카메라 이름(up/side) -> 모델이 아는 이름(camera1/camera2). camera3 은 없으면 그냥 빠짐
RENAME = {"observation.images.up": "observation.images.camera1",
          "observation.images.side": "observation.images.camera2"}


###############################################################
########## Step0 model / dataset / pre-processor ##############
###############################################################
config = PreTrainedConfig.from_pretrained(MODEL_ID)
config.device = DEVICE
policy = SmolVLAPolicy.from_pretrained(MODEL_ID, config=config)
policy.train()

# delta_timestamps: 현재 frame 부터 미래 50 step(30fps) action 을 묶어서 정답 chunk 로
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("lerobot/svla_so101_pickplace", root=DATA_DIR,
                    delta_timestamps={"action": [i / 30 for i in range(50)]})

# 정규화 stats 는 smolvla_base 것 대신 '이 데이터셋' 의 meta/stats.json 사용
stats = rename_stats(ds.meta.stats, RENAME)
pre, post = make_pre_post_processors(
    policy.config, pretrained_path=MODEL_ID,
    preprocessor_overrides={
        "device_processor": {"device": DEVICE},
        "normalizer_processor": {
            "stats": stats,
            "features": {**policy.config.input_features, **policy.config.output_features},
            "norm_map": policy.config.normalization_mapping,
        },
        "rename_observations_processor": {"rename_map": RENAME},
    },
    postprocessor_overrides={
        "unnormalizer_processor": {
            "stats": stats,
            "features": policy.config.output_features,
            "norm_map": policy.config.normalization_mapping,
        },
    },
)
print("device:", DEVICE, " frames:", len(ds))


###############################################################
########## Step1 batch 만들기 (정답 action chunk + action_is_pad)
###############################################################
loader = DataLoader(ds, batch_size=B, shuffle=True)
raw_batch = next(iter(loader))
batch = pre(rename_batch_keys(raw_batch, RENAME))

print("\n===== Step1 batch =====")
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        print(f"{k:<40} {tuple(v.shape)}")
print("action raw  [0,0]:", raw_batch[ACTION][0, 0])
print("action norm [0,0]:", batch[ACTION][0, 0])     # (a - mean) / std
print("dataset action mean:", ds.meta.stats[ACTION]["mean"])
print("dataset action std :", ds.meta.stats[ACTION]["std"])


###############################################################
########## Step2 t, noise -> x_t, u_t #########################
###############################################################
model = policy.model
actions = policy.prepare_action(batch)                      # [B, 50, 6] -> pad -> [B, 50, 32]
noise = model.sample_noise(actions.shape, actions.device)   # [B, 50, 32]
time = model.sample_time(B, actions.device)                 # [B]  Beta(1.5,1)

# 아 정답을 action 값으로 아니까, 우리는 이거를 이제 -> noise time step 마다의 velcotiy로 치환하는 작업임.
x_t, u_t, _ = make_flow_matching_inputs(actions, noise, time, convention=FlowConvention.NOISE_AT_ONE)

print("\n===== Step2 flow matching inputs =====")
print("actions:", tuple(actions.shape), " noise:", tuple(noise.shape), " t:", time)
print("x_t:", tuple(x_t.shape), " u_t:", tuple(u_t.shape))

# 손으로 검증: x_t = t*noise + (1-t)*a ,  u_t = noise - a  (= x_t 를 t 로 미분)
t3 = time[:, None, None]
print("x_t 식 일치?", torch.allclose(x_t, t3 * noise + (1 - t3) * actions))
print("u_t 식 일치?", torch.allclose(u_t, noise - actions))


###############################################################
########## Step3 loss 손으로 계산 vs policy.forward ############
###############################################################
images, img_masks = policy.prepare_images(batch)
state = policy.prepare_state(batch)
lang_tokens = batch[OBS_LANGUAGE_TOKENS]
lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

with torch.no_grad():
    # (a) VLAFlowMatching.forward : 원소별 mse [B, 50, 32]
    losses = model.forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time)
    print("\n===== Step3 loss =====")
    print("losses (elementwise):", tuple(losses.shape))

    # (b) SmolVLAPolicy.forward 가 하는 후처리를 손으로
    is_pad = batch["action_is_pad"]                        # [B, 50]
    l = losses[:, :, :6]                                   # 32 -> 6 (가짜 관절 버림)
    l = l * (~is_pad).unsqueeze(-1)                        # 에피소드 밖 step 0 처리
    num_valid = (~is_pad).sum() * 6
    my_loss = l.sum() / num_valid
    print("pad 개수:", is_pad.sum().item(), " num_valid:", num_valid.item())
    print("my loss       :", my_loss.item())

    # (c) 공식 함수 (같은 noise, time 넣어야 같은 값)
    loss, loss_dict = policy.forward(batch, noise=noise, time=time)
    print("policy.forward:", loss.item())
    print("일치?", torch.allclose(my_loss, loss))
    print(loss_dict)


###############################################################
########## Step4 backward -> 어떤 파라미터에 grad 생기나 #######
###############################################################
def group_name(n):
    # 파라미터 이름을 큰 덩어리로 묶기
    n = n.replace("model.vlm_with_expert.", "")
    if n.startswith("vlm.model.vision_model"): return "VLM vision encoder (SigLIP)"
    if n.startswith("vlm.model.connector"):    return "VLM connector"
    if n.startswith("vlm.model.text_model"):   return "VLM text model (16 layers)"
    if n.startswith("vlm.lm_head"):            return "VLM lm_head"
    if n.startswith("lm_expert"):              return "action expert"
    return n.replace("model.", "").split(".")[0]   # state_proj, action_in_proj ...

policy.zero_grad()
loss, _ = policy.forward(batch, noise=noise, time=time)   # 이번엔 no_grad 없이
loss.backward()

# 보통 모델을 만들때, named_parameters()라는 속성을 미리 만들어 놓음.
stat = defaultdict(lambda: {"n": 0, "trainable": 0, "has_grad": 0})
for n, p in policy.named_parameters():
    g = stat[group_name(n)]
    g["n"] += p.numel()
    if p.requires_grad:
        g["trainable"] += p.numel()
    if p.grad is not None and p.grad.abs().sum() > 0:
        g["has_grad"] += p.numel()

print("\n===== Step4 grad check =====")
print(f"{'group':<30} {'params':>12} {'requires_grad':>14} {'has grad':>12}")
for k, g in stat.items():
    print(f"{k:<30} {g['n']:>12,} {g['trainable']:>14,} {g['has_grad']:>12,}")
tot = sum(g["n"] for g in stat.values())
tr = sum(g["trainable"] for g in stat.values())
print(f"{'TOTAL':<30} {tot:>12,} {tr:>14,}   ({tr / tot * 100:.1f}% 학습)")
# state_proj 에 grad O = freeze 된 VLM 의 activation 을 '통과해서' 역전파됨 (state 토큰은 prefix 에 있음)


###############################################################
########## Step5 optimizer.step -> 같은 batch 로 overfit ########
###############################################################
# SmolVLA 기본 preset: AdamW(lr=1e-4, betas=(0.9,0.95), wd=1e-10), grad_clip_norm=10
opt_cfg = policy.config.get_optimizer_preset()
optimizer = opt_cfg.build(policy.get_optim_params())   # get_optim_params = policy.parameters() 전부
print("\n===== Step5 optimizer =====")
print(optimizer)

# step 전/후 비교용 스냅샷: 학습되는 W 하나, freeze 된 W 하나
w_train = policy.model.action_out_proj.weight
w_frozen = policy.model.vlm_with_expert.vlm.model.text_model.layers[0].self_attn.q_proj.weight
w_train_before, w_frozen_before = w_train.detach().clone(), w_frozen.detach().clone()

# 같은 batch, 같은 noise/time 으로 반복 -> loss 가 떨어지면 학습 루프 정상
# (warmup scheduler 를 쓰면 처음 lr 이 1e-4/1001 이라 안 떨어져 보여서 여기선 lr 고정)
N_STEPS = 20
losses_hist = []
for step in range(N_STEPS):
    optimizer.zero_grad()                                          # 1. grad 비우기
    loss, _ = policy.forward(batch, noise=noise, time=time)        # 2. forward (그래프 기록)
    loss.backward()                                                # 3. .grad 채우기
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), opt_cfg.grad_clip_norm)  # 4. 폭주 방지
    optimizer.step()                                               # 5. W 업데이트
    losses_hist.append(loss.item())
    print(f"step {step:2d}  loss {loss.item():.4f}  grad_norm {grad_norm.item():.3f}")

print("학습 W 변함?   ", not torch.equal(w_train_before, w_train.detach()))
print("freeze W 변함? ", not torch.equal(w_frozen_before, w_frozen.detach()))
print("|ΔW| mean (action_out_proj):", (w_train.detach() - w_train_before).abs().mean().item())


###############################################################
########## Step6 실제 학습의 lr 스케줄 (warmup + cosine) ########
###############################################################
# 학습 아님! 그래프로 '보여주기' 용.
# Step5 는 설명 단순하게 하려고 lr 1e-4 고정으로 돌렸지만, 실제 학습(05)은 lr 을 매 step 바꿔가며 돈다.
# 그 lr 이 어떻게 변하는지 보려고, 숫자 1개짜리 가짜 파라미터(dummy) + scheduler 만 30k 번 전진시켜서 lr 값을 기록.
# (policy / batch / loss 안 씀 -> 모델 안 건드림. 실제 적용은 05 의 scheduler.step())
dummy =torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=opt_cfg.lr)
sched = policy.config.get_scheduler_preset().build(dummy, num_training_steps=30_000)
lrs = []
for _ in range(30_000):
    lrs.append(dummy.param_groups[0]["lr"])
    dummy.step(); sched.step()
print("lr @0, 1000, 15000, 30000:", lrs[0], lrs[1000], lrs[15000], lrs[-1])

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(losses_hist, marker="o"); ax[0].set_title("overfit 1 batch (fixed noise/t)")
ax[0].set_xlabel("step"); ax[0].set_ylabel("loss")
ax[1].plot(lrs); ax[1].set_title("lr schedule: warmup 1k + cosine 30k")
ax[1].set_xlabel("step"); ax[1].set_ylabel("lr")
plt.tight_layout()
out = OUT_DIR / "04_train_loss_lr.png"
plt.savefig(out, dpi=100)
print("saved:", out)
