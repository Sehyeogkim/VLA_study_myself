'''
LIBERO 에서 GR00T fine-tune 에 넣을 데이터가 어떤 모양인지 본다.
영상 디코딩은 안 한다. meta + parquet 숫자만.
'''
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # VLA_study_myself/
DATA_DIR = ROOT / "data" / "libero"

# lerobot/libero 는 스위트 4개를 이 순서로 이어 붙인 것. 태스크 10개씩.
SUITES = [
    (0, 10, "libero_10"),       # 긴 문장, 두 동작
    (10, 20, "libero_goal"),    # 같은 씬, 목표만 다름
    (20, 30, "libero_object"),  # 같은 자리, 물건만 다름
    (30, 40, "libero_spatial"), # 검은 그릇, 위치만 다름
]

STATE_NAMES = [
    "eef_x", "eef_y", "eef_z",          # 손끝 위치 (m)
    "eef_ax", "eef_ay", "eef_az",       # 손 회전, axis-angle (rad)
    "gripper_l", "gripper_r",           # 손가락 두 개 위치
]

ACTION_NAMES = [
    "dx", "dy", "dz",                   # 손 이동량
    "dax", "day", "daz",                # 손 회전량
    "gripper",                          # -1 닫힘 / +1 열림 에 가까운 명령
]


def main():
    info = json.loads((DATA_DIR / "meta" / "info.json").read_text())
    stats = json.loads((DATA_DIR / "meta" / "stats.json").read_text())
    tasks = pd.read_parquet(DATA_DIR / "meta" / "tasks.parquet").sort_values("task_index")
    episodes = pd.read_parquet(DATA_DIR / "meta" / "episodes" / "chunk-000" / "file-000.parquet")

    print("=" * 60)
    print(f"repo          lerobot/libero")
    print(f"robot         {info['robot_type']}   fps {info['fps']}")
    print(f"episodes      {info['total_episodes']}")
    print(f"frames        {info['total_frames']}")
    print(f"tasks         {info['total_tasks']}")
    length = episodes["length"]
    print(
        f"ep length     min {int(length.min())}  median {int(length.median())}  "
        f"max {int(length.max())}  frames  "
        f"({length.min()/info['fps']:.1f}s ~ {length.max()/info['fps']:.1f}s)"
    )

    print("=" * 60)
    print("한 프레임이 GR00T 에 들어가는 칸")
    for key, feat in info["features"].items():
        if key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
            continue
        print(f"  {key:28} {feat['dtype']:8} shape {feat['shape']}")
    print("  task                         str      (instruction)")

    print("=" * 60)
    print("observation.state  8  =  손끝 xyz + 회전 3 + 그리퍼 손가락 2")
    st = stats["observation.state"]
    for i, name in enumerate(STATE_NAMES):
        print(
            f"  [{i}] {name:12} min {st['min'][i]:8.3f}  max {st['max'][i]:8.3f}  "
            f"mean {st['mean'][i]:8.3f}  std {st['std'][i]:6.3f}"
        )

    print("action  7  =  Δxyz + Δ회전 3 + 그리퍼 명령")
    ac = stats["action"]
    for i, name in enumerate(ACTION_NAMES):
        print(
            f"  [{i}] {name:12} min {ac['min'][i]:8.3f}  max {ac['max'][i]:8.3f}  "
            f"mean {ac['mean'][i]:8.3f}  std {ac['std'][i]:6.3f}"
        )

    # 에피소드마다 task 는 프레임 parquet 에만 있다. 컬럼 두 개만 읽는다.
    frames = pd.read_parquet(
        DATA_DIR / "data",
        columns=["episode_index", "task_index"],
    )
    per_ep = frames.groupby("episode_index")["task_index"].first()
    counts = per_ep.value_counts().sort_index()

    print("=" * 60)
    print("스위트 (태스크 10개 묶음)  —  fine-tune 때 여기를 통째로 쓸지, 하나만 쓸지")
    text_by_id = {int(row.task_index): str(text) for text, row in tasks.iterrows()}
    for start, end, name in SUITES:
        n_ep = int(counts.loc[start:end - 1].sum())
        print(f"\n[{name}]  tasks {start}-{end - 1}   episodes {n_ep}")
        for i in range(start, end):
            print(f"  {i:02d}  ({int(counts.loc[i]):3d} ep)  {text_by_id[i]}")

    n_parquet = len(list((DATA_DIR / "data").rglob("*.parquet")))
    n_img = len(list((DATA_DIR / "videos" / "observation.images.image").rglob("*.mp4")))
    n_img2 = len(list((DATA_DIR / "videos" / "observation.images.image2").rglob("*.mp4")))
    need_img = episodes["videos/observation.images.image/file_index"].nunique()
    need_img2 = episodes["videos/observation.images.image2/file_index"].nunique()
    print("=" * 60)
    print(f"로컬 파일   parquet {n_parquet}   image mp4 {n_img}/{need_img}   wrist mp4 {n_img2}/{need_img2}")


if __name__ == "__main__":
    main()
