"""블로그용 그림: 왜 노이즈가 '가장 가까운 정답'으로 가지 않는가.

flow matching은 노이즈 분포 전체를 데이터 분포 전체로 옮긴다.
노이즈를 각자 가장 가까운 정답으로 보내면 분포가 망가진다는 걸 비교한다.
먼저 step1_unconditional_2d.py를 실행해서 step1_moons.pt를 만들어야 한다.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib import font_manager

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from step1_unconditional_2d import VelocityMLP, sample, sample_data  # noqa: E402

font_path = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
font_manager.fontManager.addfont(font_path)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
plt.rcParams["axes.unicode_minus"] = False

model = VelocityMLP()
model.load_state_dict(torch.load(ROOT / "step1_moons.pt"))
model.eval()

torch.manual_seed(0)
N = 4000
noise = torch.randn(N, 2)

# (a) flow matching: 노이즈를 10스텝 Euler로 이동
with torch.no_grad():
    x = noise.clone()
    for i in range(10):
        x = x + 0.1 * model(x, torch.full((N, 1), i * 0.1))
flow_end = x

# (b) 가장 가까운 정답으로 보내기
data = sample_data("moons", 20000)
nearest_end = data[torch.cdist(noise, data).argmin(dim=1)]

# 도착 위치의 x좌표로 색칠 → 출발점에서 어느 쪽으로 갔는지 보인다
def colors(end):
    return end[:, 0].numpy()

LIM = 3.5
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
panels = [
    (axes[0, 0], noise, colors(flow_end), "Flow matching: 출발 (노이즈)\n색 = 도착할 위치"),
    (axes[0, 1], flow_end, colors(flow_end), "Flow matching: 도착\n반달 전체에 고르게 퍼짐"),
    (axes[1, 0], noise, colors(nearest_end), "가장 가까운 정답으로: 출발 (노이즈)\n색 = 도착할 위치"),
    (axes[1, 1], nearest_end, colors(nearest_end), "가장 가까운 정답으로: 도착\n안쪽에 몰리고 양 끝은 비어버림"),
]
data_bg = sample_data("moons", 3000).numpy()
for ax, pts, c, title in panels:
    ax.scatter(data_bg[:, 0], data_bg[:, 1], s=1, color="#e5e7eb", zorder=0)
    p = pts.numpy()
    ax.scatter(p[:, 0], p[:, 1], s=2, c=c, cmap="turbo", vmin=-3, vmax=3, alpha=0.7)
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=12)

# 양 끝(x < -2.2 또는 x > 2.2)에 도착한 비율 비교
def tip_ratio(end):
    return ((end[:, 0] < -2.2) | (end[:, 0] > 2.2)).float().mean().item()

r_data, r_flow, r_near = tip_ratio(data), tip_ratio(flow_end), tip_ratio(nearest_end)
print(f"양 끝에 도착한 비율  data={r_data:.1%}  flow={r_flow:.1%}  nearest={r_near:.1%}")
fig.suptitle(f"양 끝(|x|>2.2)에 도착한 비율 — 실제 데이터 {r_data:.0%} · flow matching {r_flow:.0%} · 가장 가까운 정답 {r_near:.0%}",
             fontsize=12, y=0.995)
fig.tight_layout()
fig.savefig(Path(__file__).parent / "step1_nearest_vs_flow.png", dpi=150,
            bbox_inches="tight", facecolor="white")
print("saved")
