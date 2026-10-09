"""블로그용 그림: ② 모델 구조와 TimeEmbedding을 step1 코드의 실제 값으로 그린다."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).parent.parent))
from step1_unconditional_2d import TimeEmbedding  # noqa: E402

font_path = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
font_manager.fontManager.addfont(font_path)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).parent
BLUE, ORANGE, GRAY, GREEN = "#2563eb", "#ea580c", "#6b7280", "#16a34a"


# ---------------------------------------------------------------------------
# 그림 1: 모델 구조
# ---------------------------------------------------------------------------
def box(ax, x, y, w, h, text, color, fs=11):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1",
                                fc=color, ec=color, alpha=0.15, lw=0))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1",
                                fc="none", ec=color, lw=1.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, p0, p1):
    ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#374151"))


fig, ax = plt.subplots(figsize=(14.5, 4.2))
ax.set_xlim(0, 14.3); ax.set_ylim(0, 4.2); ax.axis("off")

box(ax, 0.2, 2.6, 1.6, 0.9, "위치 x_t\n(2)", GRAY)
box(ax, 0.2, 0.6, 1.6, 0.9, "시간 t\n(1)", GRAY)
box(ax, 2.4, 0.6, 2.2, 0.9, "TimeEmbedding\nsin/cos → (32)", ORANGE)
arrow(ax, (1.85, 1.05), (2.35, 1.05))

box(ax, 5.2, 1.4, 1.4, 1.4, "concat\n(34)", GRAY)
arrow(ax, (1.85, 3.05), (5.15, 2.4))
arrow(ax, (4.65, 1.05), (5.15, 1.8))

layers = ["Linear\n34→256\n+ SiLU", "Linear\n256→256\n+ SiLU", "Linear\n256→256\n+ SiLU", "Linear\n256→2"]
x = 7.0
prev = 6.65
for i, name in enumerate(layers):
    box(ax, x, 1.25, 1.15, 1.7, name, BLUE, fs=10)
    arrow(ax, (prev, 2.1), (x - 0.05, 2.1))
    prev = x + 1.2
    x += 1.4
ax.text(7.0 + 1.4 * 1.5 + 0.55, 3.3, "MLP (파라미터 약 14만 개)", ha="center", fontsize=11, color=BLUE)

box(ax, 13.0, 1.6, 1.1, 1.0, "속도 v\n(2)", GREEN)
arrow(ax, (prev - 0.05, 2.1), (12.95, 2.1))

ax.text(7.1, 0.05, "(위치, 시간) → 속도.  출력 차원 = 액션 차원", ha="center", fontsize=11, color="#374151")
fig.savefig(OUT / "step1_model_architecture.png", dpi=160, bbox_inches="tight", facecolor="white")
plt.close(fig)


# ---------------------------------------------------------------------------
# 그림 2: TimeEmbedding이 t를 어떻게 펼치는가
# ---------------------------------------------------------------------------
emb = TimeEmbedding(32)
ts = torch.linspace(0, 1, 6000).unsqueeze(1)
E = emb(ts).numpy()            # (1000, 32): 앞 16개 sin, 뒤 16개 cos
freqs = emb.freqs.numpy()

fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw=dict(width_ratios=[1.1, 1]))

ax = axes[0]
im = ax.imshow(E[:, :16].T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
               extent=[0, 1, 15.5, -0.5], interpolation="nearest")
ax.set_yticks(range(16))
ax.set_yticklabels([f"{f:.0f}" if f >= 10 else f"{f:.1f}" for f in freqs], fontsize=8)
ax.set_xlabel("시간 t  (점선: t=0.30, 0.31)")
ax.set_ylabel("주파수 (위: 낮음 → 아래: 높음)")
ax.set_title("sin(t × 주파수) 값 16개 (cos 16개도 같은 패턴)", fontsize=12)
for t0 in (0.30, 0.31):
    ax.axvline(t0, color="black", lw=0.8, ls="--")
fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

ax = axes[1]
for idx, color, label in [(0, ORANGE, f"낮은 주파수 ({freqs[0]:.0f})"),
                          (13, BLUE, f"높은 주파수 ({freqs[13]:.0f})")]:
    ax.plot(ts.numpy(), E[:, idx], color=color, lw=1.4, label=label)
    for t0, mk in ((0.30, "o"), (0.31, "s")):
        v = np.sin(t0 * freqs[idx])
        ax.plot(t0, v, mk, color=color, ms=8, mec="black", zorder=5)
ax.axvspan(0.29, 0.32, color="#fde68a", alpha=0.5, zorder=0)
ax.set_xlim(0.2, 0.42)
ax.set_xlabel("시간 t  (노란 영역: t=0.30 ● vs 0.31 ■)")
ax.set_title("낮은 주파수 = 대략 언제, 높은 주파수 = 미세한 차이", fontsize=12)
ax.legend(loc="lower left", fontsize=10)

fig.tight_layout()
fig.savefig(OUT / "step1_time_embedding.png", dpi=160, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("saved")
