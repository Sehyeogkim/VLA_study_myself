"""Phase 0 / 3단계: Action chunk 생성.

액션이 점 하나 (2,) 가 아니라 H칸짜리 궤적 (H, 2) 가 된다.
  1단계:  x = (B, 2)
  3단계:  x = (B, H, 2)   # H = 16칸, 칸마다 (x, y) 경유점

학습/추론 코드는 1단계와 같다. 모델 입력/출력 크기만 커진다.
(MLP라서 (H, 2)를 펼쳐서 H*2 = 32개 숫자로 넣는다. π0는 칸 하나하나를 transformer 토큰으로 넣는다.)

데이터: 시작(0,-1) → 목표(0,1), 가운데 장애물(0,0)을 왼쪽 또는 오른쪽으로 50:50 회피하는 시연.

비교 실험: chunk 모델 vs 칸별(per-step) 모델
  - chunk:    궤적 16칸을 한 번에 생성 → 궤적 전체가 하나의 모드(왼쪽 or 오른쪽)
  - per-step: 칸마다 따로 생성 (칸 번호 T를 조건으로) → 칸마다 왼/오를 따로 골라서 지그재그

실행:
  uv run python phase0_flow_matching/step3_action_chunk.py
"""
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from step1_unconditional_2d import TimeEmbedding

FIG_DIR = Path(__file__).parent / "figures"
H = 16                      # chunk 길이 (칸 수)
OBSTACLE = (0.0, 0.0, 0.3)  # (x, y, 반지름)


# ---------------------------------------------------------------------------
# Step1. 데이터: 장애물 회피 궤적 (B, H, 2)
# ---------------------------------------------------------------------------
def sample_data(n: int) -> torch.Tensor:
    side = torch.randint(0, 2, (n, 1)).float() * 2 - 1           # -1 왼쪽, +1 오른쪽 (50:50)
    amp = 0.5 + 0.3 * torch.rand(n, 1)                           # 얼마나 크게 돌아가는지
    y = torch.linspace(-1, 1, H).expand(n, H)                    # 아래 → 위로 전진
    x = side * amp * torch.sin(math.pi * (y + 1) / 2)            # 가운데에서 가장 크게 벗어남
    traj = torch.stack([x, y], dim=-1)                           # (n, H, 2)
    return traj + 0.02 * torch.randn_like(traj)


# ---------------------------------------------------------------------------
# Step2. 모델
# ---------------------------------------------------------------------------
class ChunkVelocityMLP(nn.Module):
    """궤적 전체 (H, 2)를 받아서 궤적 전체의 속도 (H, 2)를 낸다."""

    def __init__(self, hidden=512, t_dim=32):
        super().__init__()
        self.t_embed = TimeEmbedding(t_dim)
        self.net = nn.Sequential(
            nn.Linear(H * 2 + t_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, H * 2),
        )

    def forward(self, x, t):                       # x: (B, H, 2)
        B = x.shape[0]
        h = torch.cat([x.reshape(B, H * 2), self.t_embed(t)], dim=-1)
        return self.net(h).reshape(B, H, 2)


class PerStepVelocityMLP(nn.Module):
    """비교용: 경유점 하나 (2,)만 보고 속도 (2,)를 낸다. 몇 번째 칸인지(k)는 조건으로 준다."""

    def __init__(self, hidden=256, t_dim=32, k_dim=32):
        super().__init__()
        self.t_embed = TimeEmbedding(t_dim)
        self.k_embed = nn.Embedding(H, k_dim)
        self.net = nn.Sequential(
            nn.Linear(2 + t_dim + k_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x, t, k):                    # x: (B, 2), k: (B,)
        return self.net(torch.cat([x, self.t_embed(t), self.k_embed(k)], dim=-1))


# ---------------------------------------------------------------------------
# Step3. 학습: loss는 1단계와 똑같다
# ---------------------------------------------------------------------------
def chunk_loss(model, x1):
    x0 = torch.randn_like(x1)                      # (B, H, 2) 노이즈 — 칸마다 랜덤
    t = torch.rand(x1.shape[0], 1)                 # 궤적 하나당 시간 t 하나 (16칸 전부 같은 t)
    tt = t[:, :, None]                             # (B, 1, 1) 로 늘려서 곱하기
    x_t = (1 - tt) * x0 + tt * x1
    target = x1 - x0
    return ((model(x_t, t) - target) ** 2).mean()


def per_step_loss(model, x1):
    B = x1.shape[0]
    k = torch.randint(0, H, (B,))                  # 궤적에서 칸 하나만 랜덤으로 뽑기
    p1 = x1[torch.arange(B), k]                    # (B, 2)
    p0 = torch.randn_like(p1)
    t = torch.rand(B, 1)
    p_t = (1 - t) * p0 + t * p1
    return ((model(p_t, t, k) - (p1 - p0)) ** 2).mean()


def train(model, loss_fn, name, steps=8000, batch_size=1024, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    losses = []
    for step in range(1, steps + 1):
        loss = loss_fn(model, sample_data(batch_size))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
        if step % 2000 == 0 or step == 1:
            print(f"[{name}] step {step:5d} | loss {np.mean(losses[-100:]):.4f}")
    return losses


# ---------------------------------------------------------------------------
# Step4. 추론
# ---------------------------------------------------------------------------
@torch.no_grad()
def sample_chunk(model, n, num_steps=10, return_traj=False):
    x = torch.randn(n, H, 2)                       # 16칸 전부 노이즈에서 출발
    dt = 1.0 / num_steps
    traj = [x.clone()]
    for i in range(num_steps):
        x = x + dt * model(x, torch.full((n, 1), i * dt))   # 16칸이 한꺼번에 이동
        traj.append(x.clone())
    return (x, torch.stack(traj)) if return_traj else x


@torch.no_grad()
def sample_per_step(model, n, num_steps=10):
    out = torch.zeros(n, H, 2)
    dt = 1.0 / num_steps
    for k in range(H):                             # 칸마다 따로, 서로 모른 채로 생성
        x = torch.randn(n, 2)
        kk = torch.full((n,), k, dtype=torch.long)
        for i in range(num_steps):
            x = x + dt * model(x, torch.full((n, 1), i * dt), kk)
        out[:, k] = x
    return out


# ---------------------------------------------------------------------------
# 평가 지표
# ---------------------------------------------------------------------------
def collision_rate(traj):
    ox, oy, r = OBSTACLE
    d = ((traj[..., 0] - ox) ** 2 + (traj[..., 1] - oy) ** 2).sqrt()
    return (d < r).any(dim=1).float().mean().item()


def mode_switch_rate(traj):
    """궤적 중간(장애물 근처 칸들)에서 왼쪽/오른쪽이 바뀐 적이 있는 비율."""
    mid = traj[:, 3:H - 3, 0]
    s = torch.sign(mid)
    return (s[:, 1:] != s[:, :-1]).any(dim=1).float().mean().item()


# ---------------------------------------------------------------------------
# 시각화
# ---------------------------------------------------------------------------
def draw_world(ax, title):
    ox, oy, r = OBSTACLE
    ax.add_patch(plt.Circle((ox, oy), r, color="#111827", zorder=3))
    ax.plot(0, -1, "o", color="#16a34a", ms=8, zorder=4)
    ax.plot(0, 1, "*", color="#16a34a", ms=13, zorder=4)
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.25, 1.25)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=11)


def plot_paths(ax, traj, color, n=20, alpha=0.7):
    for p in traj[:n].numpy():
        ax.plot(p[:, 0], p[:, 1], "-o", color=color, lw=1.2, ms=2.2, alpha=alpha, zorder=2)


def plot_all(chunk_model, step_model):
    FIG_DIR.mkdir(exist_ok=True)

    # (a) 디노이징 과정: 노이즈 궤적 → 매끄러운 궤적
    torch.manual_seed(1)
    _, traj = sample_chunk(chunk_model, 6, return_traj=True)  # (11, 6, H, 2)
    snaps = [0, 3, 6, 10]
    fig, axes = plt.subplots(2, len(snaps), figsize=(4 * len(snaps), 8),
                             gridspec_kw=dict(height_ratios=[1.15, 1]))
    colors = plt.cm.tab10(np.arange(6))
    for col, s in enumerate(snaps):
        ax = axes[0, col]
        draw_world(ax, f"τ = {s / 10:.1f}" + ("  (노이즈)" if s == 0 else "  (완성)" if s == 10 else ""))
        for j in range(6):
            p = traj[s, j].numpy()
            ax.plot(p[:, 0], p[:, 1], "-o", color=colors[j], lw=1.2, ms=2.5, zorder=2)
        ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)   # 모든 τ에서 같은 축 범위
        # 아래 줄: 보내주신 그림처럼 가로축 = 칸 번호 T, 세로축 = x 좌표 (궤적 하나만)
        ax = axes[1, col]
        ax.plot(np.arange(1, H + 1), traj[s, 0, :, 0].numpy(), "-o", color=colors[0], ms=4)
        ax.axhline(0, color="#9ca3af", lw=0.8, ls="--")
        ax.set_ylim(-2.5, 2.5)
        ax.set_xlabel("T (칸 번호)")
        if col == 0:
            ax.set_ylabel("x 좌표 (궤적 1개)")
    fig.suptitle("Action chunk 디노이징: 16칸이 한꺼번에 노이즈 → 매끄러운 궤적", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "step3_denoising.png", dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # (b) chunk vs per-step
    torch.manual_seed(0)
    data = sample_data(2000)
    chunk = sample_chunk(chunk_model, 2000)
    per = sample_per_step(step_model, 2000)
    stats = {}
    for name, tr in [("data", data), ("chunk", chunk), ("per-step", per)]:
        stats[name] = (collision_rate(tr), mode_switch_rate(tr))
        print(f"{name:9s} | 충돌 {stats[name][0]:6.1%} | 중간에 좌우 바뀜 {stats[name][1]:6.1%}")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    for ax, (name, tr, color, title) in zip(axes, [
        ("data", data, "#6b7280", "시연 데이터 (왼 50% / 오 50%)"),
        ("chunk", chunk, "#2563eb", "Chunk: 16칸 한꺼번에 생성"),
        ("per-step", per, "#dc2626", "Per-step: 칸마다 따로 생성"),
    ]):
        draw_world(ax, title)
        plot_paths(ax, tr, color)
        c, m = stats[name]
        ax.text(0, -1.2, f"충돌 {c:.0%} · 좌우 바뀜 {m:.0%}", ha="center", fontsize=10,
                bbox=dict(fc="white", ec="#d1d5db"))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "step3_chunk_vs_perstep.png", dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    from matplotlib import font_manager
    font_path = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
    font_manager.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams["axes.unicode_minus"] = False

    torch.manual_seed(0)
    chunk_model = ChunkVelocityMLP()
    step_model = PerStepVelocityMLP()
    print(f"chunk params: {sum(p.numel() for p in chunk_model.parameters()):,}")
    train(chunk_model, chunk_loss, "chunk")
    train(step_model, per_step_loss, "per-step")
    torch.save(chunk_model.state_dict(), Path(__file__).parent / "step3_chunk.pt")
    torch.save(step_model.state_dict(), Path(__file__).parent / "step3_per_step.pt")
    plot_all(chunk_model, step_model)
    print("figures saved: figures/step3_*.png")


if __name__ == "__main__":
    main()
