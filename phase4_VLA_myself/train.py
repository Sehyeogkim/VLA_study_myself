
import torch
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from configuration import Config
from utills import UtilsForVLA
# from modeling import SmolVLA      # 아직 없음 (encoder + expert 묶는 최상위 클래스 만들 때 다시)

'''
    GPT said:
    이번엔 Dataset 클래스를 직접 만들 필요가 거의 없어요. 
    LeRobotDataset이 이미 Dataset이에요(__len__, __getitem__이 다 있어요). 
    우리가 정할 건 "무엇을, 어떤 모양으로 꺼낼지"뿐이에요.
'''

# def train_main():
def train_main(overfit=False, steps=None):
    '''
    overfit=True : batch 1 개만 계속 학습 -> loss 가 0 근처로 떨어지는지 (버그 체크, 몇 분)
    steps        : training_steps 덮어쓰기 (None 이면 config 값)
    '''

    # Step0. Enviroment Setup
    cfg = Config()
    if steps is not None:
        cfg.training_steps = steps
    if overfit:
        cfg.warmup_steps = 20                      # 짧게 돌리니까 warmup 도 짧게
        cfg.val_every = cfg.training_steps + 1     # overfit 은 val / 저장 안 함 (train loss 만 봄)
    ROOT = Path(__file__).resolve().parents[1]          # VLA_study_myself/
    DATA_DIR = ROOT / "data" / "pusht"                  # 이미 받아둔 로컬 데이터 (다운로드 X)

    # 데이터셋에서 시간에 따른 데이터들을 한묶음으로 묶어야함.
    # 현재 시간과 이전 프레임의 사진이 필요하고, 현재 시간부터 미래까지의 action chunck가 필요하다.
    delta_timestamps = {
        "observation.image":[-0.1,0.0],
        "action":[a / cfg.fps for a in range(cfg.chunk_size)]     # 0.0, 0.1, ..., 1.5 (fps / chunk 는 config 에서)
    }


    # overall 206 episodes, 190 for train, 16 for validation
    train_ds = LeRobotDataset("lerobot/pusht", root=DATA_DIR, delta_timestamps=delta_timestamps, episodes=list(range(0, 190)))   # 인자 이름은 episodes
    val_ds = LeRobotDataset("lerobot/pusht", root=DATA_DIR, delta_timestamps=delta_timestamps, episodes=list(range(190, 206)))


    # Step2. DataLoader
    # train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    # val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    # num_workers: mp4 디코딩을 여러 CPU 에서 / drop_last: 마지막 작은 batch 버림
    # val: shuffle + 고정 seed -> val episode 여러 곳에서 고르게 뽑되 매번 같은 20 batch
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                                               num_workers=cfg.num_workers, drop_last=True, persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=True,
                                             num_workers=cfg.num_workers, generator=torch.Generator().manual_seed(0))

    from itertools import islice
    # val_batches = list(islice(val_loader, 20))
    val_batches = list(islice(val_loader, cfg.val_batches))

    # 데이터 확인용: batch 하나 꺼내서 key / shape 출력 (모델 붙이기 전)
    b = next(iter(train_loader))
    for k, v in b.items():
        print(k, tuple(v.shape) if torch.is_tensor(v) else (type(v).__name__, len(v)), flush=True)

    # # Step3. Call SMolVLA_overall for the loss
    from modeling import SmolVLA_overall
    model = SmolVLA_overall(cfg, train_ds.meta.stats).to(cfg.device)

    # batch 안 tensor 만 GPU 로 (task 는 str list 라 그대로)
    def to_device(batch):
        return {k: (v.to(cfg.device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    # loss = model(to_device(b))
    # print("loss", loss.item())
    with torch.no_grad():
        print("loss (step 0, 랜덤 expert -> 약 2 근처면 정상)", model(to_device(b)).item(), flush=True)

    # # Step4. Optimizer
    # optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    # lr 두 그룹: 새로 만든 부분은 크게, pretrained (SigLIP, SmolLM2) 는 작게 (크게 건드리면 원래 능력 잊음)
    pretrained = list(model.encoder.img_encoder.parameters()) + list(model.encoder.lm.parameters())
    pretrained_ids = {id(p) for p in pretrained}
    new = [p for p in model.parameters() if id(p) not in pretrained_ids]     # expert 전체 + mm/state projector
    print(f"params: new {sum(p.numel() for p in new):,} (lr {cfg.lr_new}) | "
          f"pretrained {sum(p.numel() for p in pretrained):,} (lr {cfg.lr_pretrained})", flush=True)
    optimizer = torch.optim.AdamW([
        {"params": new, "lr": cfg.lr_new},
        {"params": pretrained, "lr": cfg.lr_pretrained},
    ])
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(optimizer, cfg.warmup_steps, cfg.training_steps)
    params = [p for g in optimizer.param_groups for p in g["params"]]

    def val_loss():
        # flow matching loss 는 noise / t 를 랜덤으로 뽑음 -> seed 고정해서 매번 같은 문제로 채점
        model.eval()
        g = torch.random.get_rng_state(); gc = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        torch.manual_seed(0)
        total = 0.0
        with torch.no_grad():
            for vb in val_batches:
                total += model(to_device(vb)).item()
        torch.random.set_rng_state(g)
        if gc is not None: torch.cuda.set_rng_state(gc)
        model.train()
        return total / len(val_batches)

    # # Step5. Train
    # for epoch in range(cfg.epochs):
    #     pass       # 비어 있으면 SyntaxError -> 모델 만들 때 채움
    # step 단위 학습 (phase3 와 같음): DataLoader 를 다 쓰면 (StopIteration) 새 epoch
    import time, json
    OUT = Path(__file__).parent / "result"
    OUT.mkdir(exist_ok=True)
    best = float("inf")
    hist = {"train_loss": [], "val_loss": []}
    t0 = time.time()
    it = iter(train_loader)
    model.train()

    for step in range(1, cfg.training_steps + 1):
        if overfit:
            batch = b                              # 맨 처음 꺼낸 batch 하나만 반복
        else:
          try:
            batch = next(it)
          except StopIteration:
            it = iter(train_loader)
            batch = next(it)

        loss = model(to_device(batch))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        if step % cfg.log_every == 0:
            hist["train_loss"].append((step, loss.item()))
            print(f"step {step} | loss {loss.item():.4f} | lr {scheduler.get_last_lr()[0]:.2e} | "
                  f"{step / (time.time() - t0):.2f} it/s", flush=True)

        if step % cfg.val_every == 0 or step == cfg.training_steps:
            v = val_loss(); hist["val_loss"].append((step, v))
            print(f"step {step} | val {v:.4f}", flush=True)
            if v < best:
                best = v
                # buffer (정규화 mean/std) 까지 state_dict 에 들어감 -> 추론 때 그대로 load
                torch.save({"model": model.state_dict(), "step": step, "val": v}, OUT / "best.pt")
            json.dump(hist, open(OUT / "history.json", "w"))

    return


if __name__ == "__main__":
    # train_main()
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--overfit", action="store_true")   # python train.py --overfit --steps 300
    ap.add_argument("--steps", type=int, default=None)
    args = ap.parse_args()
    train_main(args.overfit, args.steps)