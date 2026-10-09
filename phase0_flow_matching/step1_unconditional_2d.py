"""Phase 0 / 1단계: 2D 무조건부 Flow Matching.

목표: 노이즈 분포 N(0, I)를 2D 데이터 분포(두 개의 달 등)로 옮기는 속도장 v_θ(x, t)를 학습한다.

  학습
    x0 ~ N(0, I)                  # 노이즈
    x1 ~ 데이터                    # 정답
    t  ~ U(0, 1)
    x_t = (1 - t) * x0 + t * x1    # 노이즈와 데이터를 잇는 직선 위의 점
    목표 속도 u = x1 - x0           # 그 직선을 따라 움직이는 속도 (t와 무관하게 일정)
    loss = || v_θ(x_t, t) - u ||²

  샘플링 (Euler 적분)
    x ← N(0, I)에서 시작
    t = 0 → 1 까지 N스텝 동안  x ← x + (1/N) * v_θ(x, t)

시간 규칙: t=0 노이즈, t=1 데이터. (openpi/π0 코드는 반대 방향이니 Phase 2에서 주의)

실행:
  uv run python phase0_flow_matching/step1_unconditional_2d.py --dataset moons
"""
import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

FIG_DIR = Path(__file__).parent / "figures"


# ---------------------------------------------------------------------------
# 1. 데이터: 2D 장난감 분포
# ---------------------------------------------------------------------------
def sample_data(name: str, n: int) -> torch.Tensor:
    if name == "moons":
        # 두 개의 반달. 모드가 2개인 분포.
        n1 = n // 2
        a = torch.rand(n1) * math.pi
        b = torch.rand(n - n1) * math.pi
        upper = torch.stack([torch.cos(a), torch.sin(a)], dim=1)
        lower = torch.stack([1 - torch.cos(b), 0.5 - torch.sin(b)], dim=1)
        x = torch.cat([upper, lower]) + 0.05 * torch.randn(n, 2)
        x = (x - torch.tensor([0.5, 0.25])) * 2.0  # 대략 [-3, 3] 범위로
    elif name == "8gaussians":
        # 원 위의 8개 가우시안. 모드가 8개인 분포.
        k = torch.randint(0, 8, (n,))
        angle = k.float() * (2 * math.pi / 8)
        centers = 2.5 * torch.stack([torch.cos(angle), torch.sin(angle)], dim=1)
        x = centers + 0.15 * torch.randn(n, 2)
    elif name == "checkerboard":
        # 체스판 패턴. 모서리가 날카로워 학습이 더 어렵다.
        x1 = torch.rand(n) * 4 - 2
        x2 = torch.rand(n) - torch.randint(0, 2, (n,)).float() * 2
        x2 = x2 + (torch.floor(x1) % 2)
        x = torch.stack([x1, x2], dim=1) * 1.5
    else:
        raise ValueError(name)
    return x[torch.randperm(n)]


# ---------------------------------------------------------------------------
# 2. 모델: v_θ(x_t, t) → 속도 (2D)
# ---------------------------------------------------------------------------
class TimeEmbedding(nn.Module):
    """스칼라 t를 sin/cos 특징 벡터로 바꾼다. π0도 시간 t를 같은 방식으로 임베딩한다."""

    def __init__(self, dim: int = 32):
        super().__init__()
        self.register_buffer("freqs", torch.exp(torch.linspace(0, math.log(1000), dim // 2)))

    def forward(self, t: torch.Tensor) -> torch.Tensor:  # t: (B, 1)
        args = t * self.freqs  # (B, dim/2)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class VelocityMLP(nn.Module):
    def __init__(self, x_dim: int = 2, hidden: int = 256, t_dim: int = 32):
        super().__init__()
        self.t_embed = TimeEmbedding(t_dim)
        self.net = nn.Sequential(
            nn.Linear(x_dim + t_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, x_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, self.t_embed(t)], dim=-1))


# ---------------------------------------------------------------------------
# 3. 학습: flow matching loss
# ---------------------------------------------------------------------------
def flow_matching_loss(model: nn.Module, x1: torch.Tensor) -> torch.Tensor:
    x0 = torch.randn_like(x1)                 # 노이즈
    t = torch.rand(x1.shape[0], 1)            # 각 샘플마다 다른 시간
    x_t = (1 - t) * x0 + t * x1               # 직선 경로 위의 점
    target = x1 - x0                          # 목표 속도
    pred = model(x_t, t)
    return ((pred - target) ** 2).mean()


def train(model, dataset, steps, batch_size, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    losses = []
    for step in range(1, steps + 1):
        loss = flow_matching_loss(model, sample_data(dataset, batch_size))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
        if step % 1000 == 0 or step == 1:
            print(f"step {step:5d} | loss {np.mean(losses[-100:]):.4f}")
    return losses


# ---------------------------------------------------------------------------
# 4. 샘플링: Euler 방법으로 ODE 적분
# ---------------------------------------------------------------------------
@torch.no_grad()
def sample(model, n, num_steps, return_traj=False):
    x = torch.randn(n, 2)                     # t=0: 순수 노이즈
    dt = 1.0 / num_steps
    traj = [x.clone()]
    for i in range(num_steps):
        t = torch.full((n, 1), i * dt)
        x = x + dt * model(x, t)              # 속도 방향으로 dt만큼 이동
        traj.append(x.clone())
    return (x, torch.stack(traj)) if return_traj else x


# ---------------------------------------------------------------------------
# 5. 시각화
# ---------------------------------------------------------------------------
def scatter(ax, x, title, color="#2563eb", lim=3.5):
    x = x.numpy()
    ax.scatter(x[:, 0], x[:, 1], s=1.5, alpha=0.5, color=color)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=11)


def plot_all(model, dataset, losses, out_prefix):
    FIG_DIR.mkdir(exist_ok=True)

    # (a) 데이터 vs 생성 샘플 + loss 곡선
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    scatter(axes[0], sample_data(dataset, 5000), "Data (target)", color="#6b7280")
    scatter(axes[1], sample(model, 5000, num_steps=10), "Generated (Euler, 10 steps)")
    axes[2].plot(np.convolve(losses, np.ones(100) / 100, mode="valid"))
    axes[2].set_title("Training loss (moving avg 100)", fontsize=11)
    axes[2].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_samples.png", dpi=140)
    plt.close(fig)

    # (b) 적분 스텝 수에 따른 품질 변화
    step_list = [1, 2, 5, 10, 50]
    fig, axes = plt.subplots(1, len(step_list), figsize=(3.2 * len(step_list), 3.4))
    torch.manual_seed(0)
    for ax, k in zip(axes, step_list):
        scatter(ax, sample(model, 5000, num_steps=k), f"{k} step{'s' if k > 1 else ''}")
    fig.suptitle("Sample quality vs. number of Euler steps", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_num_steps.png", dpi=140)
    plt.close(fig)

    # (c) 노이즈 → 데이터 궤적 (중간 시점 스냅샷 + 개별 점의 이동 경로)
    torch.manual_seed(1)
    _, traj = sample(model, 2000, num_steps=50, return_traj=True)  # (51, n, 2)
    snaps = [0, 10, 25, 40, 50]
    fig, axes = plt.subplots(1, len(snaps) + 1, figsize=(3.2 * (len(snaps) + 1), 3.4))
    for ax, s in zip(axes, snaps):
        scatter(ax, traj[s], f"t = {s / 50:.1f}")
    ax = axes[-1]
    paths = traj[:, :60].numpy()
    for j in range(paths.shape[1]):
        ax.plot(paths[:, j, 0], paths[:, j, 1], color="#2563eb", lw=0.7, alpha=0.6)
    ax.scatter(paths[0, :, 0], paths[0, :, 1], s=6, color="#9ca3af", zorder=3, label="start (noise)")
    ax.scatter(paths[-1, :, 0], paths[-1, :, 1], s=6, color="#dc2626", zorder=3, label="end (data)")
    ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Trajectories", fontsize=11)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_trajectory.png", dpi=140)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="moons", choices=["moons", "8gaussians", "checkerboard"])
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    model = VelocityMLP()
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")
    losses = train(model, args.dataset, args.steps, args.batch_size, args.lr)

    ckpt = Path(__file__).parent / f"step1_{args.dataset}.pt"
    torch.save(model.state_dict(), ckpt)
    print(f"model saved: {ckpt}")

    out_prefix = FIG_DIR / f"step1_{args.dataset}"
    plot_all(model, args.dataset, losses, out_prefix)
    print(f"figures saved: {out_prefix}_*.png")


if __name__ == "__main__":
    main()
