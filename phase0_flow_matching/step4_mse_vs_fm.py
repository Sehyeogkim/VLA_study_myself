"""Phase 0 / 4단계: MSE 회귀 vs Flow Matching.

같은 시연 데이터로 두 정책을 학습해서 비교한다.
  - 회귀:          관측 → 궤적 16칸 (한 번에 정답 하나)
  - Flow matching: 관측 + 노이즈 궤적 + τ → 속도 (10스텝 반복)

관측 = 장애물의 x 위치 ox.
  - 장애물이 한쪽으로 치우치면 시연은 넓은 쪽으로만 돌아간다 → 정답 하나
  - 장애물이 가운데 근처면 시연이 왼쪽/오른쪽으로 반반 갈린다 → 정답 두 개
회귀는 정답이 두 개인 곳에서만 망가진다는 걸 보여주는 게 목표.

실행:
  uv run python phase0_flow_matching/step4_mse_vs_fm.py
"""
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from step1_unconditional_2d import TimeEmbedding

FIG_DIR = Path(__file__).parent / "figures"
H = 16
R = 0.3          # 장애물 반지름
OX_MAX = 0.5     # 장애물 x 위치 범위 [-0.5, 0.5]


# ---------------------------------------------------------------------------
# 데이터: 장애물 위치(관측) + 회피 궤적(정답)
# ---------------------------------------------------------------------------
def sample_data(n: int):
    ox = (torch.rand(n, 1) * 2 - 1) * OX_MAX
    p_right = torch.sigmoid(-ox / 0.05)                  # 장애물이 왼쪽에 있을수록 오른쪽으로 돈다. 가운데면 50:50
    side = torch.where(torch.rand(n, 1) < p_right, 1.0, -1.0)
    peak = ox + side * (R + 0.2 + 0.2 * torch.rand(n, 1))  # 장애물 옆을 지나는 x 위치
    y = torch.linspace(-1, 1, H).expand(n, H)
    x = peak * torch.sin(math.pi * (y + 1) / 2)          # 시작/끝은 x=0, 가운데에서 peak
    traj = torch.stack([x, y], dim=-1) + 0.02 * torch.randn(n, H, 2)
    return ox, traj


def obs_encoder(dim=32):
    """관측(장애물 위치) → 벡터. VLA에서 VLM이 하는 역할을 아주 작게 흉내낸 것."""
    return nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))


# ---------------------------------------------------------------------------
# 모델 두 개
# ---------------------------------------------------------------------------
class RegressionPolicy(nn.Module):
    """관측 → 궤적. 노이즈도 τ도 없다. 같은 관측이면 항상 같은 궤적."""

    def __init__(self, hidden=512):
        super().__init__()
        self.obs = obs_encoder()
        self.net = nn.Sequential(
            nn.Linear(32, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, H * 2),
        )

    def forward(self, ox):
        return self.net(self.obs(ox)).reshape(-1, H, 2)


class FlowPolicy(nn.Module):
    """관측 + 노이즈 섞인 궤적 + τ → 궤적의 속도. 3단계 chunk 모델에 관측만 추가."""

    def __init__(self, hidden=512):
        super().__init__()
        self.obs = obs_encoder()
        self.t_embed = TimeEmbedding(32)
        self.net = nn.Sequential(
            nn.Linear(H * 2 + 32 + 32, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, H * 2),
        )

    def forward(self, x, t, ox):
        B = x.shape[0]
        h = torch.cat([x.reshape(B, H * 2), self.t_embed(t), self.obs(ox)], dim=-1)
        return self.net(h).reshape(B, H, 2)


# ---------------------------------------------------------------------------
# 학습
# ---------------------------------------------------------------------------
def regression_loss(model, ox, x1):
    return ((model(ox) - x1) ** 2).mean()          # 시연 궤적을 그대로 맞히기


def flow_loss(model, ox, x1):
    x0 = torch.randn_like(x1)
    t = torch.rand(x1.shape[0], 1)
    tt = t[:, :, None]
    x_t = (1 - tt) * x0 + tt * x1
    return ((model(x_t, t, ox) - (x1 - x0)) ** 2).mean()


def train(model, loss_fn, name, steps=8000, batch_size=1024, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    for step in range(1, steps + 1):
        ox, x1 = sample_data(batch_size)
        loss = loss_fn(model, ox, x1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 2000 == 0 or step == 1:
            print(f"[{name}] step {step:5d} | loss {loss.item():.4f}")


# ---------------------------------------------------------------------------
# 추론
# ---------------------------------------------------------------------------
@torch.no_grad()
def sample_flow(model, ox, num_steps=10):
    x = torch.randn(ox.shape[0], H, 2)
    for i in range(num_steps):
        x = x + (1 / num_steps) * model(x, torch.full((ox.shape[0], 1), i / num_steps), ox)
    return x


@torch.no_grad()
def sample_regression(model, ox):
    return model(ox)


def collision(traj, ox):
    d = ((traj[..., 0] - ox) ** 2 + traj[..., 1] ** 2).sqrt()   # 장애물 중심 (ox, 0)
    return (d < R).any(dim=1).float()


# ---------------------------------------------------------------------------
# 시각화
# ---------------------------------------------------------------------------
GRAY, RED, BLUE = "#9ca3af", "#dc2626", "#2563eb"


def draw_world(ax, ox, title):
    ax.add_patch(plt.Circle((ox, 0), R, color="#111827", zorder=3))
    ax.plot(0, -1, "o", color="#16a34a", ms=8, zorder=4)
    ax.plot(0, 1, "*", color="#16a34a", ms=13, zorder=4)
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=11)


def demos_at(ox_value, n):
    """특정 장애물 위치에서의 시연 궤적 n개."""
    ox, traj = sample_data(20000)
    idx = (ox[:, 0] - ox_value).abs().argsort()[:n]
    return traj[idx]


def plot_paths(ax, traj, color, lw=1.3, alpha=0.7, zorder=2):
    for p in traj.numpy():
        ax.plot(p[:, 0], p[:, 1], "-o", color=color, lw=lw, ms=2, alpha=alpha, zorder=zorder)


def plot_all(reg, flow):
    FIG_DIR.mkdir(exist_ok=True)
    torch.manual_seed(0)

    # (a) 대표 그림: 장애물 정중앙, 회귀 vs flow matching
    ox0 = torch.zeros(12, 1)
    demo = demos_at(0.0, 16)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2))
    for ax, name in zip(axes, ["reg", "flow"]):
        plot_paths(ax, demo, GRAY, lw=1, alpha=0.5, zorder=1)
        if name == "reg":
            draw_world(ax, 0.0, "MSE 회귀: 두 시연의 평균 → 장애물로 직진")
            plot_paths(ax, sample_regression(reg, ox0[:1]), RED, lw=3, alpha=1, zorder=5)
        else:
            draw_world(ax, 0.0, "Flow matching: 매번 한쪽을 골라 회피")
            plot_paths(ax, sample_flow(flow, ox0), BLUE, lw=1.8, alpha=0.85, zorder=5)
    axes[0].plot([], [], color=GRAY, label="사람 시연 (왼 50% / 오 50%)")
    axes[0].plot([], [], color=RED, lw=3, label="회귀 예측 (항상 같은 궤적)")
    axes[1].plot([], [], color=GRAY, label="사람 시연 (왼 50% / 오 50%)")
    axes[1].plot([], [], color=BLUE, lw=2, label="flow matching 샘플 12개")
    for ax in axes:
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.16), fontsize=10, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "step4_hero.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # (b) 장애물 위치별 비교
    oxs = [-0.4, -0.1, 0.0, 0.1, 0.4]
    fig, axes = plt.subplots(2, len(oxs), figsize=(3.3 * len(oxs), 6.8))
    for col, v in enumerate(oxs):
        demo = demos_at(v, 12)
        o = torch.full((10, 1), v)
        for row, (name, color) in enumerate([("회귀", RED), ("Flow", BLUE)]):
            ax = axes[row, col]
            plot_paths(ax, demo, GRAY, lw=1, alpha=0.45, zorder=1)
            if row == 0:
                tr = sample_regression(reg, o[:1])
                plot_paths(ax, tr, color, lw=3, alpha=1, zorder=5)
            else:
                tr = sample_flow(flow, o)
                plot_paths(ax, tr, color, lw=1.6, alpha=0.85, zorder=5)
            draw_world(ax, v, f"{name} · 장애물 x = {v:+.1f}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "step4_by_position.png", dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # (c) 장애물 위치별 충돌률
    grid = torch.linspace(-OX_MAX, OX_MAX, 41)
    reg_rate, flow_rate, demo_split = [], [], []
    for v in grid:
        o = torch.full((400, 1), v.item())
        reg_rate.append(collision(sample_regression(reg, o), o).mean().item())
        flow_rate.append(collision(sample_flow(flow, o), o).mean().item())
        p = torch.sigmoid(-v / 0.05).item()
        demo_split.append(min(p, 1 - p) * 2)   # 1 = 50:50, 0 = 한쪽만
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.fill_between(grid.numpy(), 0, np.array(demo_split), color="#fde68a", alpha=0.6,
                    label="시연이 갈리는 정도 (1 = 왼/오 50:50)")
    ax.plot(grid.numpy(), reg_rate, "-o", color=RED, ms=3, label="MSE 회귀 충돌률")
    ax.plot(grid.numpy(), flow_rate, "-o", color=BLUE, ms=3, label="Flow matching 충돌률")
    ax.set_xlabel("장애물 x 위치 (관측)")
    ax.set_ylabel("비율")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title("정답이 두 개로 갈리는 곳에서만 회귀가 무너진다", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "step4_collision_curve.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # 숫자 요약
    o_mid = torch.zeros(2000, 1)
    o_side = torch.cat([torch.full((1000, 1), -0.4), torch.full((1000, 1), 0.4)])
    for name, fn in [("regression", lambda o: sample_regression(reg, o)), ("flow", lambda o: sample_flow(flow, o))]:
        mid = collision(fn(o_mid), o_mid).mean().item()
        side = collision(fn(o_side), o_side).mean().item()
        print(f"{name:10s} | 충돌률  장애물 가운데 {mid:6.1%}  |  장애물 치우침(±0.4) {side:6.1%}")


def main():
    from matplotlib import font_manager
    font_path = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
    font_manager.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams["axes.unicode_minus"] = False

    torch.manual_seed(0)
    reg, flow = RegressionPolicy(), FlowPolicy()
    train(reg, regression_loss, "regression")
    train(flow, flow_loss, "flow")
    torch.save(reg.state_dict(), Path(__file__).parent / "step4_regression.pt")
    torch.save(flow.state_dict(), Path(__file__).parent / "step4_flow.pt")
    plot_all(reg, flow)
    print("figures saved: figures/step4_*.png")


if __name__ == "__main__":
    main()
