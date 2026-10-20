'''
check the data in the LIBERO
'''
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROOT = Path(__file__).resolve().parents[1]  # VLA_study_myself/
DATA_DIR = ROOT / "data" / "libero"

ds = LeRobotDataset(
    "lerobot/libero",
    root=DATA_DIR,
    episodes=[0],
    download_videos=False,
    video_backend="pyav",
)

print(ds)
item = ds[0]
for k, v in item.items():
    if hasattr(v, "shape"):
        print(f"{k}: {tuple(v.shape)} {v.dtype}")
    else:
        print(f"{k}: {type(v).__name__} {v}")
