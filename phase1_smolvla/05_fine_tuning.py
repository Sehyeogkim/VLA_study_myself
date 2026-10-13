'''
05 Fine-tuning SmolVLA - lerobot-train 의 학습 루프를 직접 풀어쓴 버전

04 에서는 "1 step" 을 해부했다면, 여기서는 그걸 수천 step 반복해서 진짜로 학습시킨다.

[전체 흐름]
  Step0  설정         steps / batch / lr preset / 출력 폴더
  Step1  데이터 분리   에피소드 0~44 = train,  45~49 = val (학습에 한 번도 안 쓰는 에피소드)
  Step2  모델/전처리   smolvla_base 에서 시작 + 이 데이터셋 stats 로 정규화
  Step3  optimizer    AdamW(1e-4) + warmup/cosine scheduler (steps 에 맞춰 자동 스케일)
  
  # Main
  Step4  학습 루프     (04 Step5 와 동일) zero_grad -> forward -> backward -> clip(10) -> step -> sched.step
                      - LOG_EVERY  : train loss / grad_norm / lr / 속도
                      - EVAL_EVERY : val loss (고정 noise/t 로 계산해서 step 간 비교 가능)
                      - SAVE_EVERY : checkpoint (policy + pre/post processor)
  
  # Eval
  Step5  평가         val 에피소드에서 action chunk 예측 vs 정답 (실제 단위, 관절별 MAE)
                      학습 전(base) vs 학습 후 비교 그래프
  결과 -> results/05_*/
[실행]
  RunPod(GPU) 권장. 로컬 CPU 는 너무 느림.
  python phase1_smolvla/05_fine_tuning.py --steps 2000 --batch_size 32
'''
import argparse
import csv
import itertools
import time as timer
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from lerobot.configs import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.processor.rename_processor import rename_batch_keys, rename_stats
from lerobot.utils.constants import ACTION


###############################################################
########## Step0 설정 ##########################################
###############################################################
p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=2000)
p.add_argument("--batch_size", type=int, default=32)
p.add_argument("--num_workers", type=int, default=4)
p.add_argument("--log_every", type=int, default=50)
p.add_argument("--eval_every", type=int, default=250)
p.add_argument("--save_every", type=int, default=1000)
p.add_argument("--seed", type=int, default=1000)
args = p.parse_args()

ROOT = Path(__file__).resolve().parents[1]          # VLA_study_myself/
DATA_DIR = ROOT / "data" / "svla_so101_pickplace"
OUT_DIR = Path(__file__).resolve().parent / "results" / f"05_finetune_{args.steps}steps"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "lerobot/smolvla_base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK = 50
FPS = 30
RENAME = {"observation.images.up": "observation.images.camera1",
          "observation.images.side": "observation.images.camera2"}

torch.manual_seed(args.seed)
print(f"device={DEVICE}  steps={args.steps}  batch={args.batch_size}  out={OUT_DIR}")


###############################################################
########## Step1 데이터: train / val 에피소드 분리 ##############
###############################################################

# total 50 Episodes or videos.
delta = {"action": [i / FPS for i in range(CHUNK)]}          # 미래 50 step 정답 chunk
TRAIN_EPS = list(range(0, 45))
VAL_EPS = list(range(45, 50))

train_ds = LeRobotDataset("lerobot/svla_so101_pickplace", root=DATA_DIR, episodes=TRAIN_EPS, delta_timestamps=delta)
val_ds = LeRobotDataset("lerobot/svla_so101_pickplace", root=DATA_DIR, episodes=VAL_EPS, delta_timestamps=delta)
print(f"train frames={len(train_ds)}  val frames={len(val_ds)}")


# Train Data Loader.
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                          num_workers=args.num_workers, pin_memory=DEVICE == "cuda",
                          persistent_workers=args.num_workers > 0)
# 주의: worker 는 메인 프로세스가 영상을 디코딩하기 '전에' 만들어야 함.
# 메인에서 torchcodec 으로 먼저 디코딩한 뒤 fork 하면 worker 가 디코더 상태를 물려받아서
# "Could not push packet to decoder" 에러가 남 (lerobot-train 은 spawn 으로 회피)
train_iter = iter(train_loader)
# val 은 고정된 batch 몇 개만 (매번 같은 frame -> step 간 비교 가능)
val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(0), num_workers=0)
val_raw_batches = list(itertools.islice(val_loader, 4))


def next_batch():
    # 에폭 경계 없이 계속 batch 를 뽑기 (lerobot_train 도 이렇게 cycle 함)
    global train_iter
    try:
        return next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)                   # persistent worker 재사용
        return next(train_iter)


###############################################################
########## Step2 모델 + pre/post processor #####################
###############################################################
config = PreTrainedConfig.from_pretrained(MODEL_ID)
config.device = DEVICE
policy = SmolVLAPolicy.from_pretrained(MODEL_ID, config=config)

# 정규화 stats = 이 데이터셋 (train 에피소드 기준 아니라 전체 stats.json, lerobot-train 과 동일)
stats = rename_stats(train_ds.meta.stats, RENAME)
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


def prepare(raw):
    return pre(rename_batch_keys(dict(raw), RENAME))


val_batches = [prepare(b) for b in val_raw_batches]
# val loss 용 고정 noise / t  (랜덤이면 val loss 가 출렁여서 비교가 안 됨)
g = torch.Generator(device="cpu").manual_seed(123)
val_noise = [torch.randn(b[ACTION].shape[0], CHUNK, policy.config.max_action_dim, generator=g).to(DEVICE)
             for b in val_batches]
val_time = [(torch.rand(b[ACTION].shape[0], generator=g) * 0.999 + 0.001).to(DEVICE) for b in val_batches]


@torch.no_grad()
def val_loss():
    vals = [policy.forward(b, noise=n, time=t)[0].item() for b, n, t in zip(val_batches, val_noise, val_time)]
    return sum(vals) / len(vals)


@torch.no_grad()
def predict_val(tag):
    """val batch 0 에서 action chunk 예측 (10 step denoising) -> 실제 단위로 GT 와 비교"""
    policy.reset()                                         # action queue 비우기
    b = val_batches[0]
    pred = post(policy.predict_action_chunk(b))            # [B, 50, 6] 역정규화 (도 단위). post 가 cpu 로 옮김
    gt = val_raw_batches[0][ACTION].to(pred.device)        # [B, 50, 6] 원본 값
    valid = ~b["action_is_pad"].to(pred.device)            # [B, 50]
    mae = ((pred - gt).abs() * valid[..., None]).sum(dim=(0, 1)) / valid.sum()
    print(f"[{tag}] val chunk MAE per joint:", [round(x, 2) for x in mae.tolist()],
          " mean:", round(mae.mean().item(), 2))
    policy.train()
    return pred[0].cpu(), gt[0].cpu(), mae.mean().item()


###############################################################
########## Step3 optimizer + scheduler (SmolVLA preset) ########
###############################################################
opt_cfg = policy.config.get_optimizer_preset()          # AdamW lr=1e-4, betas=(0.9,0.95), clip=10
optimizer = opt_cfg.build(policy.get_optim_params())
# steps < 30k 이면 warmup/decay 를 비율대로 줄여줌 (ex. 2000 step -> warmup 66, decay 2000)
scheduler = policy.config.get_scheduler_preset().build(optimizer, num_training_steps=args.steps)

n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
print(f"trainable params: {n_train:,}")


###############################################################
########## Step4 학습 루프 #####################################
###############################################################
pred_before, gt0, mae_before = predict_val("before")     # 학습 전 base 모델 성능
history = []                                             # (step, train_loss, val_loss, lr, grad_norm)
v0 = val_loss()
print(f"step     0  val {v0:.4f}")
history.append((0, None, v0, scheduler.get_last_lr()[0], None))

policy.train()
t0 = timer.time()
for step in range(1, args.steps + 1):
    batch = prepare(next_batch())

    optimizer.zero_grad()
    loss, _ = policy.forward(batch)                       # 랜덤 noise / t (실제 학습)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), opt_cfg.grad_clip_norm)
    optimizer.step()
    scheduler.step()                                      # lr 갱신 (warmup -> cosine)

    lr = scheduler.get_last_lr()[0]
    vl = None
    if step % args.eval_every == 0 or step == args.steps:
        vl = val_loss()
    if step % args.log_every == 0 or vl is not None:
        sps = step / (timer.time() - t0)
        eta = (args.steps - step) / sps / 60
        msg = f"step {step:5d}  loss {loss.item():.4f}  grad {grad_norm.item():6.2f}  lr {lr:.2e}  {sps:.2f} it/s  eta {eta:.1f}m"
        print(msg + (f"  | val {vl:.4f}" if vl is not None else ""))
    history.append((step, loss.item(), vl, lr, grad_norm.item()))

    if step % args.save_every == 0 or step == args.steps:
        ckpt = OUT_DIR / "checkpoints" / f"{step:06d}"
        policy.save_pretrained(ckpt)                      # config.json + model.safetensors
        pre.save_pretrained(ckpt)                         # 정규화 stats / rename / tokenizer 설정
        post.save_pretrained(ckpt)
        print("saved checkpoint:", ckpt)

print(f"train time: {(timer.time() - t0) / 60:.1f} min")


###############################################################
########## Step5 평가: 학습 전 vs 후 ###########################
###############################################################
pred_after, _, mae_after = predict_val("after")

with open(OUT_DIR / "history.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["step", "train_loss", "val_loss", "lr", "grad_norm"])
    w.writerows(history)

steps_ = [h[0] for h in history if h[1] is not None]
tr_ = [h[1] for h in history if h[1] is not None]
vs_ = [(h[0], h[2]) for h in history if h[2] is not None]

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].plot(steps_, tr_, alpha=0.4, label="train loss")
ax[0].plot([s for s, _ in vs_], [v for _, v in vs_], "o-", label="val loss (fixed noise/t)")
ax[0].set_yscale("log"); ax[0].set_xlabel("step"); ax[0].legend(); ax[0].set_title("loss")
ax[1].plot([h[0] for h in history], [h[3] for h in history]); ax[1].set_title("lr"); ax[1].set_xlabel("step")
plt.tight_layout(); plt.savefig(OUT_DIR / "loss_lr.png", dpi=100); plt.close()

names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
fig, ax = plt.subplots(2, 3, figsize=(15, 7))
for j, a in enumerate(ax.flatten()):
    a.plot(gt0[:, j], "k-", lw=2, label="GT")
    a.plot(pred_before[:, j], "r--", label=f"before (MAE {mae_before:.1f})")
    a.plot(pred_after[:, j], "b-", label=f"after (MAE {mae_after:.1f})")
    a.set_title(names[j]); a.set_xlabel("chunk step")
ax[0, 0].legend(fontsize=8)
plt.suptitle("val episode: predicted action chunk vs GT (unnormalized)")
plt.tight_layout(); plt.savefig(OUT_DIR / "pred_vs_gt.png", dpi=100); plt.close()
print("saved:", OUT_DIR / "loss_lr.png", OUT_DIR / "pred_vs_gt.png", OUT_DIR / "history.csv")
