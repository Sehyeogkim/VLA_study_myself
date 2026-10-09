"""블로그용 그림: 방식 A(회귀) vs 방식 B(생성 모델)로 행동을 뽑을 때의 차이."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyBboxPatch

font_path = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
font_manager.fontManager.addfont(font_path)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(0)
BLUE, ORANGE, RED, GRAY = "#2563eb", "#ea580c", "#dc2626", "#9ca3af"


def demo_path(side, n=60, noise=0.0):
    """시작점(0,0)에서 목표(0,10)까지, 장애물(0,5)을 side 방향(+1 오른쪽/-1 왼쪽)으로 회피."""
    y = np.linspace(0, 10, n)
    x = side * 2.2 * np.sin(np.pi * y / 10) ** 2
    x += noise * rng.standard_normal() * np.sin(np.pi * y / 10)
    return x, y


def draw_box(ax, xy, w, h, text, color):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec="none", alpha=0.15))
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc="none", ec=color, lw=1.8))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=12)


def arrow(ax, x0, x1, y):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#374151"))


def draw_scene(ax, title):
    ax.add_patch(Circle((0, 5), 1.0, color="#111827"))
    ax.text(0, 3.55, "장애물", color="#111827", ha="center", va="center", fontsize=10, zorder=5, bbox=dict(fc="white", ec="none", pad=1))
    ax.plot(0, 0, "o", color="#16a34a", ms=10)
    ax.text(0.35, -0.2, "시작", fontsize=10)
    ax.plot(0, 10, "*", color="#16a34a", ms=16)
    ax.text(0.4, 10, "목표", fontsize=10, va="center")
    ax.set_xlim(-4, 4); ax.set_ylim(-1, 11)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=13, pad=8)


fig = plt.figure(figsize=(12, 10.5))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 2.6], hspace=0.12, wspace=0.08)

# ---- 윗줄: 구조 도식 ----
axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1])
for ax in (axA, axB):
    ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")

axA.set_title("방식 A: 회귀 (MSE)", fontsize=15, fontweight="bold", color=RED)
draw_box(axA, (0.2, 1.0), 2.0, 1.0, "관측 o", GRAY)
arrow(axA, 2.3, 3.6, 1.5)
draw_box(axA, (3.7, 1.0), 2.6, 1.0, "네트워크\n(1번 실행)", RED)
arrow(axA, 6.4, 7.7, 1.5)
draw_box(axA, (7.8, 1.0), 2.0, 1.0, "관절값 a\n(항상 같음)", GRAY)
axA.text(5, 0.3, "정답 하나를 맞히도록 학습 → 정답들의 '평균'을 출력",
         ha="center", fontsize=11, color="#374151")

axB.set_title("방식 B: 생성 모델 (Flow Matching)", fontsize=15, fontweight="bold", color=BLUE)
draw_box(axB, (0.2, 1.6), 2.0, 0.8, "관측 o", GRAY)
draw_box(axB, (0.2, 0.55), 2.0, 0.8, "랜덤 노이즈", GRAY)
arrow(axB, 2.3, 3.6, 1.5)
draw_box(axB, (3.7, 1.0), 2.6, 1.0, "네트워크\n(10번 반복)", BLUE)
axB.annotate("", xy=(5.6, 2.1), xytext=(4.4, 2.1),
             arrowprops=dict(arrowstyle="-|>", lw=1.4, color=BLUE,
                             connectionstyle="arc3,rad=-0.6"))
arrow(axB, 6.4, 7.7, 1.5)
draw_box(axB, (7.8, 1.0), 2.0, 1.0, "관절값 a\n(매번 다름)", GRAY)
axB.text(5, 0.3, "정답들의 '분포'를 학습 → 그중 하나를 샘플링",
         ha="center", fontsize=11, color="#374151")

# ---- 아랫줄: 장애물 회피 결과 ----
sA = fig.add_subplot(gs[1, 0]); sB = fig.add_subplot(gs[1, 1])

for ax in (sA, sB):
    for side in (-1, 1):
        for _ in range(8):
            x, y = demo_path(side, noise=0.25)
            ax.plot(x, y, color=GRAY, lw=1, alpha=0.6, zorder=1)

draw_scene(sA, "결과: 왼쪽/오른쪽 시연의 평균 → 장애물로 직진 (실패)")
xl, y = demo_path(-1); xr, _ = demo_path(1)
sA.plot((xl + xr) / 2, y, color=RED, lw=4, zorder=3, label="회귀 모델의 예측")
sA.text(0, 7.8, "충돌!", color=RED, fontsize=16, fontweight="bold", ha="center",
        bbox=dict(fc="white", ec=RED, boxstyle="round,pad=0.3"), zorder=4)

draw_scene(sB, "결과: 매번 한쪽 경로를 골라 회피 (성공)")
for i, side in enumerate([-1, 1, -1, 1, 1, -1]):
    x, y = demo_path(side, noise=0.2)
    sB.plot(x, y, color=BLUE, lw=2.5, alpha=0.85, zorder=3,
            label="생성 모델의 샘플" if i == 0 else None)

for ax in (sA, sB):
    ax.plot([], [], color=GRAY, lw=1, label="사람 시연 (왼쪽 50% / 오른쪽 50%)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.1), fontsize=10, frameon=False, ncol=1)

out = __file__.replace(".py", ".png")
fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
print(out)
