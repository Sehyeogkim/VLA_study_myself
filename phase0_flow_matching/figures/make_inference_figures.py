"""블로그용 그림: Step4 추론. 학습된 속도장(화살표 지도)과 그 위를 따라가는 Euler 10스텝.

먼저 step1_unconditional_2d.py를 실행해서 step1_moons.pt를 만들어야 한다.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from step1_unconditional_2d import VelocityMLP, sample_data  # noqa: E402

font_path = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
font_manager.fontManager.addfont(font_path)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
plt.rcParams["axes.unicode_minus"] = False

model = VelocityMLP()
model.load_state_dict(torch.load(ROOT / "step1_moons.pt"))
model.eval()

NUM_STEPS = 10
dt = 1.0 / NUM_STEPS
LIM = 3.2
COLORS = ["#dc2626", "#2563eb", "#16a34a", "#9333ea"]

# 출발점 4개 고정 (노이즈에서 뽑았다고 생각)
starts = torch.tensor([[-1.2, 1.0], [0.8, -0.6], [-0.3, -1.4], [1.5, 1.2]])


@torch.no_grad()
def euler_path(x):
    path = [x.clone()]
    for i in range(NUM_STEPS):
        t = torch.full((x.shape[0], 1), i * dt)
        x = x + dt * model(x, t)
        path.append(x.clone())
    return torch.stack(path)  # (NUM_STEPS+1, n, 2)


paths = euler_path(starts.clone()).numpy()

# 숫자로 따라가기용 출력 (점 하나)
print("점 하나의 10스텝 이동:")
with torch.no_grad():
    x = starts[:1].clone()
    for i in range(NUM_STEPS):
        t = torch.full((1, 1), i * dt)
        v = model(x, t)
        print(f"  step {i}: t={i*dt:.1f}  x=({x[0,0]:+.2f}, {x[0,1]:+.2f})  v=({v[0,0]:+.2f}, {v[0,1]:+.2f})")
        x = x + dt * v
    print(f"  final : t=1.0  x=({x[0,0]:+.2f}, {x[0,1]:+.2f})")

# ---------------------------------------------------------------------------
# 그림: 시간별 속도장 + Euler 경로
# ---------------------------------------------------------------------------
g = np.linspace(-LIM, LIM, 17)
GX, GY = np.meshgrid(g, g)
grid = torch.tensor(np.stack([GX.ravel(), GY.ravel()], 1), dtype=torch.float32)
data = sample_data("moons", 3000).numpy()

snap_steps = [0, 3, 6, 9]  # t = 0.0, 0.3, 0.6, 0.9
fig, axes = plt.subplots(1, len(snap_steps), figsize=(4.3 * len(snap_steps), 4.6))
for ax, k in zip(axes, snap_steps):
    t = k * dt
    with torch.no_grad():
        V = model(grid, torch.full((grid.shape[0], 1), t)).numpy()
    ax.scatter(data[:, 0], data[:, 1], s=1, color="#d1d5db", zorder=0)
    ax.quiver(GX, GY, V[:, 0].reshape(GX.shape), V[:, 1].reshape(GX.shape),
              color="#6b7280", alpha=0.8, angles="xy", scale_units="xy", scale=6, width=0.004)
    for j, c in enumerate(COLORS):
        ax.plot(paths[: k + 1, j, 0], paths[: k + 1, j, 1], "-", color=c, lw=1.8)
        ax.plot(paths[: k + 1, j, 0], paths[: k + 1, j, 1], "o", color=c, ms=3)
        ax.plot(paths[k, j, 0], paths[k, j, 1], "o", color=c, ms=9, mec="black", zorder=5)
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"step {k}  (t = {t:.1f})", fontsize=13)
fig.suptitle("회색 화살표 = 모델이 알려주는 속도 (시간마다 바뀜) · 색깔 점 = 노이즈 4개가 화살표를 따라 이동",
             fontsize=13, y=1.0)
fig.tight_layout()
fig.savefig(Path(__file__).parent / "step1_inference_velocity_field.png", dpi=150,
            bbox_inches="tight", facecolor="white")
plt.close(fig)

# 최종 위치 (t = 1.0)
fig, ax = plt.subplots(figsize=(4.6, 4.6))
ax.scatter(data[:, 0], data[:, 1], s=1, color="#d1d5db", zorder=0)
for j, c in enumerate(COLORS):
    ax.plot(paths[:, j, 0], paths[:, j, 1], "-o", color=c, lw=1.8, ms=3)
    ax.plot(paths[0, j, 0], paths[0, j, 1], "s", color=c, ms=8, mec="black", zorder=5)
    ax.plot(paths[-1, j, 0], paths[-1, j, 1], "*", color=c, ms=16, mec="black", zorder=5)
ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM)
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax.set_title("10스텝 후 (t = 1.0)  ■ 출발 → ★ 도착", fontsize=13)
fig.savefig(Path(__file__).parent / "step1_inference_final.png", dpi=150,
            bbox_inches="tight", facecolor="white")
plt.close(fig)
print("saved")
