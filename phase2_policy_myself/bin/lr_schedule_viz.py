"""
lr_schedule_viz.py — Full_script_train.py 의 lr_lambda (warmup + cosine) 를 눈으로 보기

LambdaLR: 매 step 마다  lr = LR * lr_lambda(step)
  1) warmup  (0 ~ WARMUP)      : lr_lambda = (s+1)/WARMUP          -> 0.05, 0.10, ... 1.0  (직선으로 올라감)
  2) cosine  (WARMUP ~ STEPS)  : lr_lambda = r + (1-r)*0.5*(1+cos(pi*progress))  -> 1.0 에서 r 로 부드럽게 내려감

실행: uv run --project /Users/jeff/project/lerobot python phase2_policy_myself/bin/lr_schedule_viz.py
"""
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch

plt.rcParams["font.family"] = "AppleGothic"   # 한글 표시 (mac)
plt.rcParams["axes.unicode_minus"] = False

LR = 1e-4
STEPS = 200
WARMUP = 20
R = 0.025                      # min_lr_ratio (끝 lr = LR * 0.025)


def lr_lambda(s):
    if s < WARMUP:
        return (s + 1) / WARMUP
    progress = (s - WARMUP) / max(1, STEPS - WARMUP)
    return R + (1 - R) * 0.5 * (1 + math.cos(math.pi * progress))


# 실제 torch LambdaLR 로 돌려서 값 기록 (손으로 계산한 거랑 같은지 확인)
dummy = torch.nn.Parameter(torch.zeros(1))
opt = torch.optim.AdamW([dummy], lr=LR)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
lrs = []
for s in range(STEPS):
    lrs.append(opt.param_groups[0]["lr"])   # 이번 step 에 쓰이는 lr
    opt.step()
    sched.step()
assert all(abs(lrs[s] - LR * lr_lambda(s)) < 1e-12 for s in range(STEPS))   # lr = LR * lr_lambda(s)

steps = list(range(STEPS))
ratio = [lr_lambda(s) for s in steps]

fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# (1) lr_lambda (배율)
a = ax[0]
a.plot(steps, ratio, lw=2)
a.axvspan(0, WARMUP, color="orange", alpha=0.15, label=f"warmup (0~{WARMUP})")
a.axvspan(WARMUP, STEPS, color="blue", alpha=0.05, label=f"cosine ({WARMUP}~{STEPS})")
a.axhline(R, color="gray", ls="--", lw=1)
a.text(STEPS * 0.98, R + 0.02, f"min ratio {R}", ha="right", color="gray")
a.set_title("lr_lambda(step)  (배율, 0~1)")
a.set_xlabel("step"); a.set_ylabel("ratio")
a.legend(loc="center right")

# (2) 실제 lr = LR * lr_lambda
a = ax[1]
a.plot(steps, lrs, lw=2, color="tab:green")
a.axvspan(0, WARMUP, color="orange", alpha=0.15)
mid = WARMUP + (STEPS - WARMUP) // 2
for s, note in [(0, "시작: 아주 작게"), (WARMUP - 1, "warmup 끝 = LR"), (mid, "cosine 절반"), (STEPS - 1, "끝: LR x 0.025")]:
    a.scatter([s], [lrs[s]], color="red", zorder=3)
    a.annotate(f"step {s}\n{lrs[s]:.1e}\n{note}", (s, lrs[s]),
               textcoords="offset points", xytext=(10, 10), fontsize=8)
a.set_title(f"actual lr = {LR:.0e} x lr_lambda")
a.set_xlabel("step"); a.set_ylabel("lr")

plt.tight_layout()
out = Path(__file__).resolve().parents[1] / "results" / "lr_schedule.png"
out.parent.mkdir(exist_ok=True)
plt.savefig(out, dpi=100)
print("saved:", out)
for s in [0, 1, WARMUP - 1, WARMUP, mid, STEPS - 1]:
    print(f"step {s:3d} | ratio {lr_lambda(s):.3f} | lr {lrs[s]:.2e}")
