'''
train.py — LIBERO 로 GR00T N1.7 스타일 action head 를 맞춘다.

VLM(Cosmos-Reason2-2B)은 기본으로 얼린다. 맞는 것은 DiT.
DiT 가중치는 NVIDIA 체크포인트가 아니라 여기서 랜덤으로 시작한다.
(구조가 AlternateVLDiT 와 같아서 공식을 흉내 낸 것이지, 가중치가 호환되지는 않는다.)

실행은 RunPod. 로컬에서 이 파일을 돌리면 cuda 가 없어서 바로 선다.
    cd phase7_GR00T
    python train.py --suite libero_spatial --steps 10000
    python train.py --overfit --steps 300          # batch 하나 loss 가 떨어지는지
'''
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from configuration import Config
from modeling import GrootPolicy

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "libero"

# lerobot/libero 가 스위트를 이어 붙인 순서. 태스크 10개씩.
SUITES = {
    "libero_10": (0, 10),
    "libero_goal": (10, 20),
    "libero_object": (20, 30),
    "libero_spatial": (30, 40),
}


def delta_timestamps(cfg: Config):
    '''지금 프레임의 사진/state + 앞으로 40칸(4초) action.'''
    return {
        "observation.images.image": [0.0],
        "observation.images.image2": [0.0],
        "observation.state": [0.0],
        "action": [i / cfg.fps for i in range(cfg.chunk_size)],
    }


def episode_split(suite: str | None, val_per_task: int):
    '''태스크마다 맨 끝 데모 val_per_task 개를 빼고 나머지를 학습.'''
    frames = pd.read_parquet(DATA_DIR / "data", columns=["episode_index", "task_index"])
    per_ep = frames.groupby("episode_index")["task_index"].first()
    if suite is None:
        task_lo, task_hi = 0, 40
    else:
        task_lo, task_hi = SUITES[suite]

    train_eps, val_eps = [], []
    for task_id, group in per_ep.groupby(per_ep):
        if not (task_lo <= int(task_id) < task_hi):
            continue
        eps = sorted(int(i) for i in group.index.tolist())
        k = min(val_per_task, max(1, len(eps) // 5))
        val_eps.extend(eps[-k:])
        train_eps.extend(eps[:-k])
    return train_eps, val_eps


def to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def train_main(suite=None, overfit=False, steps=None, train_backbone=False):
    cfg = Config()
    if steps is not None:
        cfg.training_steps = steps
    if train_backbone:
        cfg.train_backbone = True
    if overfit:
        cfg.warmup_steps = 20
        cfg.val_every = cfg.training_steps + 1
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda 없음. 학습은 RunPod 에서 돌린다.")

    train_eps, val_eps = episode_split(suite, cfg.val_per_task)
    print(f"suite {suite or 'all'} | train episodes {len(train_eps)} | val episodes {len(val_eps)}", flush=True)

    delta = delta_timestamps(cfg)
    train_ds = LeRobotDataset(
        "lerobot/libero", root=DATA_DIR, episodes=train_eps,
        delta_timestamps=delta, download_videos=False, video_backend="pyav",
    )
    val_ds = LeRobotDataset(
        "lerobot/libero", root=DATA_DIR, episodes=val_eps,
        delta_timestamps=delta, download_videos=False, video_backend="pyav",
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, drop_last=True,
        persistent_workers=cfg.num_workers > 0, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers,
        generator=torch.Generator().manual_seed(0), pin_memory=True,
    )
    from itertools import islice
    val_batches = list(islice(val_loader, cfg.val_batches))

    batch0 = next(iter(train_loader))
    for k, v in batch0.items():
        print(k, tuple(v.shape) if torch.is_tensor(v) else (type(v).__name__, len(v)), flush=True)

    policy = GrootPolicy(cfg, train_ds.meta.stats).to(cfg.device)
    n_dit = sum(p.numel() for p in policy.dit.parameters())
    n_vlm = sum(p.numel() for p in policy.backbone.parameters())
    print(f"params: DiT {n_dit:,} (lr {cfg.lr_dit}, train) | "
          f"VLM {n_vlm:,} ({'lr ' + str(cfg.lr_backbone) if cfg.train_backbone else 'frozen'})", flush=True)

    groups = [{"params": policy.dit.parameters(), "lr": cfg.lr_dit}]
    if cfg.train_backbone:
        groups.append({"params": policy.backbone.parameters(), "lr": cfg.lr_backbone})
    optimizer = torch.optim.AdamW(groups)
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(optimizer, cfg.warmup_steps, cfg.training_steps)
    clip_params = [p for g in optimizer.param_groups for p in g["params"]]

    import copy
    ema_dit = copy.deepcopy(policy.dit).eval()
    for p in ema_dit.parameters():
        p.requires_grad_(False)

    @torch.no_grad()
    def ema_update(step):
        decay = min(cfg.ema_decay, (1 + step) / (10 + step))
        for pe, p in zip(ema_dit.parameters(), policy.dit.parameters()):
            pe.lerp_(p, 1 - decay)

    def val_loss(dit=None):
        was = policy.training
        policy.eval()
        swapped = dit is not None
        if swapped:
            live = policy.dit
            policy.dit = dit
        g = torch.random.get_rng_state()
        gc = torch.cuda.get_rng_state()
        torch.manual_seed(0)
        total = 0.0
        with torch.no_grad():
            for vb in val_batches:
                total += policy(to_device(vb, cfg.device)).item()
        torch.random.set_rng_state(g)
        torch.cuda.set_rng_state(gc)
        if swapped:
            policy.dit = live
        if was:
            policy.train()
        return total / len(val_batches)

    OUT = Path(__file__).parent / "result"
    OUT.mkdir(exist_ok=True)
    best, best_ema = float("inf"), float("inf")
    hist = {"train_loss": [], "val_loss": [], "val_loss_ema": [], "suite": suite or "all"}
    t0 = time.time()
    it = iter(train_loader)
    policy.train()

    def save(dit, path, step, val):
        torch.save({
            "dit": dit.state_dict(),
            "state_mean": policy.state_mean.detach().cpu(),
            "state_std": policy.state_std.detach().cpu(),
            "action_mean": policy.action_mean.detach().cpu(),
            "action_std": policy.action_std.detach().cpu(),
            "step": step,
            "val": val,
            "suite": suite or "all",
        }, path)

    for step in range(1, cfg.training_steps + 1):
        if overfit:
            batch = batch0
        else:
            try:
                batch = next(it)
            except StopIteration:
                it = iter(train_loader)
                batch = next(it)

        loss = policy(to_device(batch, cfg.device))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(clip_params, cfg.grad_clip)
        optimizer.step()
        scheduler.step()
        ema_update(step)

        if step % cfg.log_every == 0:
            hist["train_loss"].append((step, loss.item()))
            print(f"step {step} | loss {loss.item():.4f} | lr {scheduler.get_last_lr()[0]:.2e} | "
                  f"{step / (time.time() - t0):.2f} it/s", flush=True)

        if step % cfg.val_every == 0 or step == cfg.training_steps:
            v = val_loss()
            ve = val_loss(ema_dit)
            hist["val_loss"].append((step, v))
            hist["val_loss_ema"].append((step, ve))
            print(f"step {step} | val {v:.4f} | val_ema {ve:.4f}", flush=True)
            if v < best:
                best = v
                save(policy.dit, OUT / "best.pt", step, v)
            if ve < best_ema:
                best_ema = ve
                save(ema_dit, OUT / "best_ema.pt", step, ve)
            json.dump(hist, open(OUT / "history.json", "w"))

        if step % cfg.save_every == 0 and not overfit:
            save(policy.dit, OUT / f"step{step}.pt", step, float("nan"))
            save(ema_dit, OUT / f"step{step}_ema.pt", step, float("nan"))

    return


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default=None, choices=list(SUITES),
                    help="비우면 스위트 4개 전부. 하나만 맞출 때는 libero_spatial 부터.")
    ap.add_argument("--overfit", action="store_true")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--train_backbone", action="store_true",
                    help="VLM 도 작은 lr 로 연다. A40 한 장에는 보통 안 들어간다.")
    args = ap.parse_args()
    train_main(args.suite, args.overfit, args.steps, args.train_backbone)
