'''check_ckpt.py — best.pt 를 modeling.py 로 로드해서 val loss 재계산 (학습 로그 0.4464 근처면 OK) + sample_actions 테스트'''
# best.pt 를 modeling.py 로 로드 -> 학습 때와 같은 방식의 val batch 로 loss 재계산 (로그 0.4464 와 비교)
import sys, time, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from pathlib import Path
from torch.utils.data import DataLoader
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from modeling import load_policy, compute_loss, sample_actions
DEV = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = Path(__file__).resolve().parents[1]
prefix, expert, cfg = load_policy(ROOT / "phase2_policy_myself/results/run1/best.pt", DEV)
ds_val = LeRobotDataset("lerobot/pusht", root=ROOT / "data/pusht",
                        delta_timestamps={"action": [i / cfg.fps for i in range(cfg.chunk_size)]},
                        episodes=list(range(190, 206)))
loader_val = DataLoader(ds_val, batch_size=64, shuffle=True, num_workers=0, drop_last=True,
                        generator=torch.Generator().manual_seed(0))
g = torch.Generator().manual_seed(0)
tot, n = 0.0, 0
t0 = time.time()
for i, b in enumerate(loader_val):
    if i >= 8: break
    b = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in b.items()}
    B = b["action"].shape[0]
    noise = torch.randn(b["action"].shape, generator=g).to(DEV)
    t = (torch.distributions.Beta(cfg.time_beta_alpha, cfg.time_beta_beta).sample((B,)) * 0.999 + 0.001).to(DEV)
    with torch.no_grad():
        tot += compute_loss(prefix, expert, b, cfg, noise, t).item(); n += 1
print(f"val loss (8 batches x 64): {tot / n:.4f}   ({time.time() - t0:.0f}s)")
# sample_actions smoke test vs ground truth
b4 = {k: (v[:4].to(DEV) if torch.is_tensor(v) else v[:4]) for k, v in b.items()}
torch.manual_seed(0)
a = sample_actions(prefix, expert, b4, cfg)
print("sample_actions:", tuple(a.shape))
print("pred  [0,:4]:", a[0, :4].cpu().numpy().round(1).tolist())
print("truth [0,:4]:", b4["action"][0, :4].cpu().numpy().round(1).tolist())
print("mean abs err (pixels):", (a - b4["action"]).abs().mean().item())
