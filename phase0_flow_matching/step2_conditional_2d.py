"""Phase 0 / 2단계: 2D 조건부 Flow Matching.

1단계와 같은데 조건 c가 하나 추가된다.
  c = 0: 위쪽 반달,  c = 1: 아래쪽 반달

  모델:   v_θ(x_t, t)  →  v_θ(x_t, t, c)
  학습:   정답 x1과 함께 그 정답의 라벨 c도 뽑아서 모델에 같이 넣어준다. loss는 그대로.
  샘플링: 원하는 c를 정해서 넣어주면 그 반달 위로만 간다.

VLA에서는 c가 "이미지 + 지시문"이다. 같은 노이즈라도 지시문이 다르면 다른 액션이 나온다.

실행:
  uv run python phase0_flow_matching/step2_conditional_2d.py
"""
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from step1_unconditional_2d import TimeEmbedding

FIG_DIR = Path(__file__).parent / "figures"
NUM_CLASSES = 2
COLORS = ["#2563eb", "#ea580c"]  # c=0 위쪽 반달, c=1 아래쪽 반달


# ---------------------------------------------------------------------------
# A. 데이터: 정답 액션 + 라벨
# ---------------------------------------------------------------------------
def sample_data(n: int):
    """1단계와 같은 반달인데, 각 점이 어느 반달인지 라벨 c도 같이 돌려준다."""
    c = torch.randint(0, NUM_CLASSES, (n,))
    a = torch.rand(n) * math.pi
    upper = torch.stack([torch.cos(a), torch.sin(a)], dim=1)
    lower = torch.stack([1 - torch.cos(a), 0.5 - torch.sin(a)], dim=1)
    x = torch.where(c[:, None] == 0, upper, lower) + 0.05 * torch.randn(n, 2)
    x = (x - torch.tensor([0.5, 0.25])) * 2.0
    return x, c


# ---------------------------------------------------------------------------
# 모델: v_θ(x_t, t, c)
# ---------------------------------------------------------------------------
class ConditionalVelocityMLP(nn.Module):
    def __init__(self, x_dim=2, hidden=256, t_dim=32, c_dim=32):
        super().__init__()
        self.t_embed = TimeEmbedding(t_dim)
        self.c_embed = nn.Embedding(NUM_CLASSES, c_dim)  # 라벨 번호 → 학습되는 벡터 32개
        self.net = nn.Sequential(
            nn.Linear(x_dim + t_dim + c_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, x_dim),
        )

    def forward(self, x, t, c):
        return self.net(torch.cat([x, self.t_embed(t), self.c_embed(c)], dim=-1))


# ---------------------------------------------------------------------------
# B. 학습 재료 + loss: 1단계와 같고 c만 같이 넣는다
# ---------------------------------------------------------------------------
def flow_matching_loss(model, x1, c):
    x0 = torch.randn_like(x1)
    t = torch.rand(x1.shape[0], 1)
    x_t = (1 - t) * x0 + t * x1
    target = x1 - x0
    pred = model(x_t, t, c)                   # ← 차이는 여기 c 하나
    return ((pred - target) ** 2).mean()


def train(model, steps=8000, batch_size=1024, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    losses = []
    for step in range(1, steps + 1):
        x1, c = sample_data(batch_size)
        loss = flow_matching_loss(model, x1, c)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
        if step % 1000 == 0 or step == 1:
            print(f"step {step:5d} | loss {np.mean(losses[-100:]):.4f}")
    return losses


# ---------------------------------------------------------------------------
# 추론: 원하는 c를 넣어서 Euler 적분
# ---------------------------------------------------------------------------
@torch.no_grad()
def sample(model, c, num_steps=10, x0=None, return_traj=False):
    x = torch.randn(c.shape[0], 2) if x0 is None else x0.clone()
    dt = 1.0 / num_steps
    traj = [x.clone()]
    for i in range(num_steps):
        t = torch.full((x.shape[0], 1), i * dt)
        x = x + dt * model(x, t, c)
        traj.append(x.clone())
    return (x, torch.stack(traj)) if return_traj else x


# ---------------------------------------------------------------------------
# 시각화
# ---------------------------------------------------------------------------
def setup(ax, title, lim=3.5):
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=11)


def plot_all(model, losses):
    FIG_DIR.mkdir(exist_ok=True)
    data, data_c = sample_data(4000)
    data, data_c = data.numpy(), data_c.numpy()

    # (a) 조건별 생성 결과
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    for k in range(NUM_CLASSES):
        m = data_c == k
        axes[0].scatter(data[m, 0], data[m, 1], s=1.5, alpha=0.5, color=COLORS[k], label=f"c={k}")
    setup(axes[0], "Data (색 = 라벨 c)")
    axes[0].legend(fontsize=8, loc="lower right", markerscale=5)
    torch.manual_seed(0)
    for k in range(NUM_CLASSES):
        gen = sample(model, torch.full((3000,), k, dtype=torch.long)).numpy()
        ax = axes[1 + k]
        ax.scatter(data[:, 0], data[:, 1], s=1, color="#e5e7eb", zorder=0)
        ax.scatter(gen[:, 0], gen[:, 1], s=1.5, alpha=0.5, color=COLORS[k])
        setup(ax, f"Generated with c={k}")
    axes[3].plot(np.convolve(losses, np.ones(100) / 100, mode="valid"))
    axes[3].set_title("Training loss (moving avg 100)", fontsize=11)
    axes[3].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "step2_samples.png", dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # (b) 같은 노이즈, 다른 조건 → 다른 도착지
    torch.manual_seed(3)
    x0 = torch.randn(6, 2)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
    for k, ax in enumerate(axes):
        _, traj = sample(model, torch.full((6,), k, dtype=torch.long), x0=x0, return_traj=True)
        traj = traj.numpy()
        ax.scatter(data[:, 0], data[:, 1], s=1, color="#e5e7eb", zorder=0)
        for j in range(traj.shape[1]):
            ax.plot(traj[:, j, 0], traj[:, j, 1], "-o", color=COLORS[k], lw=1.5, ms=2.5)
            ax.plot(traj[0, j, 0], traj[0, j, 1], "s", color="#6b7280", ms=7, mec="black", zorder=5)
            ax.plot(traj[-1, j, 0], traj[-1, j, 1], "*", color=COLORS[k], ms=14, mec="black", zorder=5)
        setup(ax, f"Same noise (■), c={k}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "step2_same_noise.png", dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # (c) t=0 화살표 지도: 조건에 따라 화살표 자체가 달라진다
    g = np.linspace(-3.2, 3.2, 17)
    GX, GY = np.meshgrid(g, g)
    grid = torch.tensor(np.stack([GX.ravel(), GY.ravel()], 1), dtype=torch.float32)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    for col, (title, cs) in enumerate([("c=0 (위쪽 반달)", [0]), ("c=1 (아래쪽 반달)", [1]),
                                       ("1단계 무조건부 (참고)", None)]):
        ax = axes[col]
        ax.scatter(data[:, 0], data[:, 1], s=1, color="#e5e7eb", zorder=0)
        with torch.no_grad():
            t0 = torch.zeros(grid.shape[0], 1)
            if cs is None:
                from step1_unconditional_2d import VelocityMLP
                ckpt = Path(__file__).parent / "step1_moons.pt"
                if not ckpt.exists():
                    ax.set_axis_off(); continue
                m1 = VelocityMLP(); m1.load_state_dict(torch.load(ckpt))
                V = m1(grid, t0).numpy(); color = "#6b7280"
            else:
                V = model(grid, t0, torch.full((grid.shape[0],), cs[0], dtype=torch.long)).numpy()
                color = COLORS[cs[0]]
        ax.quiver(GX, GY, V[:, 0].reshape(GX.shape), V[:, 1].reshape(GX.shape), color=color,
                  angles="xy", scale_units="xy", scale=6, width=0.004)
        setup(ax, f"t = 0 화살표 지도: {title}", lim=3.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "step2_velocity_t0.png", dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    from matplotlib import font_manager
    font_path = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
    font_manager.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams["axes.unicode_minus"] = False

    torch.manual_seed(0)
    model = ConditionalVelocityMLP()
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")
    losses = train(model)
    torch.save(model.state_dict(), Path(__file__).parent / "step2_moons.pt")
    plot_all(model, losses)
    print("figures saved: figures/step2_*.png")


if __name__ == "__main__":
    main()
