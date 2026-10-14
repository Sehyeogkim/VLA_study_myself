'''
train.py — 학습 루프 (Full_script_train.py 의 parse_args + main 그대로)
모델은 modeling.py 에서 import
'''
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import math

from configuration import MyPolicyConfig
from modeling import Prefix, ActionExpert, compute_loss

ROOT = Path(__file__).resolve().parents[1]          # VLA_study_myself/
DATA_DIR = ROOT / "data" / "pusht"
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def parse_args():
    # 실행 옵션 (실험할 때마다 바꾸는 값). 모델 기본값은 configuration.py
    #   로컬 디버깅: --batch_size 4 --num_workers 0 --steps 200 --overfit 1
    #   RunPod     : --batch_size 64 --num_workers 4 --steps 20000 --overfit 0
    cfg = MyPolicyConfig()
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=cfg.lr)
    p.add_argument("--warmup", type=int, default=cfg.warmup_steps)   # 처음 N step lr 0 -> LR (Adam 초반 spike 방지)
    p.add_argument("--overfit", type=int, default=1)                 # 1: batch 1개 고정 반복 (버그 체크), 0: 정상 학습
    p.add_argument("--val_every", type=int, default=500)    # 몇 step 마다 val loss
    p.add_argument("--val_batches", type=int, default=4)    # val 에 쓸 고정 batch 개수
    p.add_argument("--save_every", type=int, default=2000)  # 몇 step 마다 체크포인트 저장
    p.add_argument("--out_dir", type=str, default=str(Path(__file__).parent / "results" / "run"))
    return p.parse_args()

def main():
    args = parse_args()
    cfg = MyPolicyConfig()
    BATCH_SIZE, NUM_WORKERS, STEPS = args.batch_size, args.num_workers, args.steps
    LR, WARMUP, OVERFIT = args.lr, args.warmup, bool(args.overfit)

    # Step0 : Load the dataset from the lerobot dataset. ( train / val split based on the espisode.)

    ds_train = LeRobotDataset(
        "lerobot/pusht",
        root=DATA_DIR,
        delta_timestamps={"action": [i / cfg.fps for i in range(cfg.chunk_size)]},
        episodes=  list(range(0,190))
    )

    ds_val = LeRobotDataset(
        "lerobot/pusht",
        root=DATA_DIR,
        delta_timestamps={"action": [i / cfg.fps for i in range(cfg.chunk_size)]},
        episodes=  list(range(190,206))          # 남은 16개 전부 val
    )

    # stats: dataset 이 meta/stats.json 을 이미 읽어둠 (없으면 위에서 HF 에서 다운로드된 뒤)
    stats = ds_train.meta.stats

    # (Important) Dividing the dataset into batches.
    loader_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)

    # SHuffle should be false in oroder to avoid the random access of the dataset. (수능 시험지가 같아야 함.)
    # 섞되 seed 고정 -> 16개 val episode 에서 골고루 + 매번 같은 시험지
    loader_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True,
                            generator=torch.Generator().manual_seed(0))

    

    # Step1 : Define the Batch stream and the optimizer, instance, and scheduler.
    from utils import BatchStream
    stream_train = BatchStream(loader_train)   # train iter 먼저 생성 (phase1 torchcodec fork 교훈) -> val 은 그 다음
    # Define the prefix model instance and the optimizer with the trainable parameters.
    prefix = Prefix(cfg, stats["observation.state"]["mean"], stats["observation.state"]["std"]).to(DEVICE)
    expert = ActionExpert(cfg, stats["action"]["mean"], stats["action"]["std"]).to(DEVICE)
    params = [p for p in list(prefix.parameters()) + list(expert.parameters()) if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LR)

    # Scheduler: linear warmup (0 -> LR) + cosine decay (LR -> 2.5% LR)
    #   LambdaLR 은 매 step 마다 lr = LR * lr_lambda(step) 로 바꿔줌
    def lr_lambda(s):
        if s < WARMUP:
            return (s + 1) / WARMUP                                     # 0.05, 0.10, ... 1.0
        progress = (s - WARMUP) / max(1, STEPS - WARMUP)                # 0 -> 1
        r = cfg.min_lr_ratio
        return r + (1 - r) * 0.5 * (1 + math.cos(math.pi * progress))   # 1.0 -> min_lr_ratio
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print("================================================")
    print(f"device: {DEVICE} | overfit: {OVERFIT} | steps: {STEPS} | batch: {BATCH_SIZE}")
    print(f"trainable params: {sum(p.numel() for p in params):,}")
    print("================================================")

    def to_device(b):
        return {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in b.items()}

    fixed_batch = to_device(stream_train.next()) if OVERFIT else None        # overfit: 이 batch 하나만 반복

    # val: 고정 batch + 고정 noise/t  -> 매번 "같은 시험지" 로 채점해야 step 끼리 비교 가능
    g = torch.Generator().manual_seed(0)
    val_set = []
    for i, b in enumerate(loader_val):
        if i >= args.val_batches:
            break
        b = to_device(b)

        # Pick out the action in here.
        B = b["action"].shape[0]

        # Noise and t are fixed for the whole val set.
        noise = torch.randn(b["action"].shape, generator=g).to(DEVICE)
        t = (torch.distributions.Beta(cfg.time_beta_alpha, cfg.time_beta_beta).sample((B,)) * 0.999 + 0.001).to(DEVICE)
        
        # Append the batch, noise, and t to the val set.
        val_set.append((b, noise, t))


    print(f"train frames {len(ds_train)} | val frames {len(ds_val)} | val batches {len(val_set)}")

    def val_loss():
        expert.eval()                                   # dropout off
        with torch.no_grad():
            # Batch, Noise, Time
            v = sum(compute_loss(prefix, expert, b, cfg, n, tt).item() for b, n, tt in val_set) / len(val_set)
        expert.train()
        return v

    # checkpoint: 학습한 것만 저장 (VLM 은 HF 에서 다시 로드) -> ~72MB
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    from dataclasses import asdict
    def save_ckpt(name, step, v):
        torch.save({
            "expert": expert.state_dict(),                 # action_mean/std buffer 포함
            "state_proj": prefix.state_proj.state_dict(),
            "state_mean": prefix.state_mean.cpu(),
            "state_std": prefix.state_std.cpu(),
            "cfg": asdict(cfg),
            "step": step,
            "val_loss": v,
        }, out_dir / name)

    import time
    losses, lrs, val_hist = [], [], []                  # val_hist: (step, val_loss)
    best = float("inf")
    v0 = val_loss(); val_hist.append((0, v0))
    print(f"Step    0 | val {v0:.4f}")
    t_start = time.time()

    # Learning loop.
    for step in range(1, STEPS+1):
        
        # Step2 : Prefix + 2D Mask + Position ids = preprocessing.
        batch = fixed_batch if OVERFIT else to_device(stream_train.next())
        loss = compute_loss(prefix, expert, batch, cfg)

        # Step7 : Backward pass and update the parameters.
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)   # grad 폭주 방지
        optimizer.step()
        scheduler.step()                                   # lr 다음 값으로

        # Step8 : log
        losses.append(loss.item()); lrs.append(optimizer.param_groups[0]["lr"])
        if step == 1 or step % 50 == 0:
            msg = f"Step {step:5d} | loss {loss.item():.4f} | lr {lrs[-1]:.2e} | {step / (time.time() - t_start):.2f} it/s"
            if DEVICE == "cuda":
                msg += f" | mem {torch.cuda.max_memory_allocated() / 1e9:.1f}GB"
            print(msg, flush=True)

        # Step9 : val + checkpoint
        if step % args.val_every == 0 or step == STEPS:
            v = val_loss(); val_hist.append((step, v))
            tag = ""
            if v < best:
                best = v; save_ckpt("best.pt", step, v); tag = "  <- best (saved)"
            print(f"Step {step:5d} | val {v:.4f}{tag}", flush=True)
        if step % args.save_every == 0 or step == STEPS:
            save_ckpt(f"step_{step}.pt", step, val_hist[-1][1])

    # history 저장 (그래프 다시 그리거나 노션 정리용)
    import json
    json.dump({"train_loss": losses, "lr": lrs, "val": val_hist, "best_val": best}, open(out_dir / "history.json", "w"))

    # Step10 : loss / lr 그래프 저장
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(range(1, len(losses) + 1), losses, alpha=0.5, label="train")
    vs, vv = zip(*val_hist)
    ax[0].plot(vs, vv, "o-", color="red", label="val")
    ax[0].set_yscale("log"); ax[0].set_title(f"loss (overfit={OVERFIT}, best val {best:.4f})"); ax[0].set_xlabel("step"); ax[0].legend()
    ax[1].plot(lrs); ax[1].set_title("lr (warmup + cosine)"); ax[1].set_xlabel("step")
    plt.tight_layout()
    path = out_dir / "loss_lr.png"
    plt.savefig(path, dpi=100); plt.close()
    print("saved:", path)
    return

if __name__ == "__main__":
    main()