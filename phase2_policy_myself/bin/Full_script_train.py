'''
This script is for the training loop of the action expert model.
'''
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import torch.nn as nn
import math

from configuration import MyPolicyConfig

# 모델 하이퍼파라미터는 configuration.py, 실행 옵션은 main() 의 argparse, stats 는 main() 에서 dataset 으로부터.
# -> 이 파일을 import 만 해도 (예: eval.py) 아무것도 실행/로드되지 않음
ROOT = Path(__file__).resolve().parents[1]          # VLA_study_myself/
DATA_DIR = ROOT / "data" / "pusht"
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


class Prefix(nn.Module):

    def __init__(self, cfg: MyPolicyConfig, state_mean, state_std):
        super().__init__()
        self.cfg = cfg

        from transformers import AutoModelForImageTextToText, AutoTokenizer
        self.vlm_model = AutoModelForImageTextToText.from_pretrained(cfg.vlm_name)
        self.vlm_model.eval()                    # freeze: dropout 끄기
        self.vlm_model.requires_grad_(False)     # freeze: 학습 안 함
    
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.vlm_name)

        # All the projection layers should be here.
        self.state_proj = nn.Linear(cfg.state_dim, cfg.vlm_hidden)

        # state normalize 값: buffer 로 등록 -> .to(device) 따라가고, state_dict(체크포인트)에도 저장됨
        self.register_buffer("state_mean", torch.as_tensor(state_mean, dtype=torch.float32))
        self.register_buffer("state_std", torch.as_tensor(state_std, dtype=torch.float32))

    def prefix_preprocessing(self, batch):

        # 1.  image: [B, 3, 96, 96] (0~1, 이미 CHW) -> 512 resize -> [-1, 1]
        import torch.nn.functional as F
        img = batch["observation.image"]
        S = self.cfg.image_size
        img = F.interpolate(img, size=(S, S), mode="bilinear", align_corners=False)
        img = img * 2 - 1

        with torch.no_grad():
            vis = self.vlm_model.model.vision_model(pixel_values = img).last_hidden_state   # [B, 1024, 768]
            img_tokens = self.vlm_model.model.connector(vis)                               # [B, 64, 960]
            img_tokens = img_tokens * math.sqrt(self.cfg.vlm_hidden)                                     # SmolVLA embed_prefix 스케일

        # 2. Tokenize the text
        text = batch["task"]                     # list[str] 길이 B

        #put the padding on there
        lang = self.tokenizer(text, return_tensors ="pt", padding = "max_length", truncation = True, max_length = self.cfg.max_lang_len)
        input_ids = lang.input_ids.to(img.device)          # tokenizer 출력은 CPU -> 모델 device 로
        lang_pad = lang.attention_mask.to(img.device)
        with torch.no_grad():
            lang_tokens = self.vlm_model.model.text_model.embed_tokens(input_ids)       # [B, 30, 960]
            lang_tokens = lang_tokens * math.sqrt(self.cfg.vlm_hidden)                               # SmolVLA embed_prefix 스케일


        #3. State.
        state_t = batch["observation.state"].float()           # 이미 tensor [B, 2] -> torch.tensor() 다시 감싸면 warning
        state_n = (state_t - self.state_mean) / self.state_std

        #4. State projection.
        state_token = self.state_proj(state_n).unsqueeze(1) # understood

        #5. Prefix.

        prefix = torch.cat([img_tokens, lang_tokens, state_token], dim = 1)


        #6. Padding & Attention mask.
        B = prefix.shape[0]                      # 전역 BATCH_SIZE 대신 실제 batch 크기 사용
        dev = prefix.device
        img_pad = torch.ones(B, img_tokens.shape[1], dtype=torch.bool, device=dev)
        lang_pad_b = lang_pad.bool()
        state_pad = torch.ones(B, state_token.shape[1], dtype=torch.bool, device=dev)

        pad_masks = torch.cat([img_pad, lang_pad_b, state_pad], dim=1)
        
        att_list = ([0] * img_tokens.shape[1] + [0] * lang_tokens.shape[1] + [1] * state_token.shape[1])
        att_masks = torch.tensor(att_list, dtype=torch.bool, device=dev).unsqueeze(0).expand(B, -1)
        
        from utils import make_att_2d_masks
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        
        # 7. Position ids.
        position_ids = torch.cumsum(pad_masks, dim = 1) -1

        return prefix, att_2d_masks, position_ids, pad_masks

    def forward(self, prefix_tokens, att_2d_masks, position_ids):

        # Call the VLM model.
        out = self.vlm_model.model.text_model(
            inputs_embeds=prefix_tokens, # [B, L = 95, D = 960]
            attention_mask=att_2d_masks[:, None, :, :], # [B, 95, 95] -> [B, 1, 95, 95]  4D 여야 HF 가 그대로 사용
            position_ids=position_ids,   # [B, L = 95]
        )

        return out

class ActionExpert(nn.Module):

    def __init__(self, cfg: MyPolicyConfig, action_mean, action_std):
        super().__init__()
        self.cfg = cfg
        D = cfg.expert_hidden
        self.action_in_proj = nn.Linear(cfg.action_dim, D)
        self.action_time_mlp = nn.Sequential(nn.Linear(2 * D, D), nn.SiLU(), nn.Linear(D, D))
        self.action_out_proj = nn.Linear(D, cfg.action_dim)

        self.memory_proj = nn.Linear(cfg.vlm_hidden, D)

        # action normalize 값: 학습(normalize) / 추론(unnormalize) 모두 같은 값 쓰도록 buffer 로 저장
        self.register_buffer("action_mean", torch.as_tensor(action_mean, dtype=torch.float32))
        self.register_buffer("action_std", torch.as_tensor(action_std, dtype=torch.float32))
        
        # Define the decoer layer first and then put 4 om the overall TransformerDecoder.
        self.expert_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=D, 
                                    nhead=cfg.expert_heads, 
                                    dim_feedforward=4 * D,
                                    batch_first=True, 
                                    norm_first=True),
                                    
                                    num_layers=cfg.expert_layers)    
    
    def flow_matching(self, batch, noise=None, t=None):
        '''
        정답 action (x_0) -> 학습 재료 (x_t, t, u_t) 만들기
        action: [B, 16, 2] (normalize 된 정답 chunk)

        x_t = t * noise + (1 - t) * action     # t=0 이면 action, t=1 이면 noise
        u_t = d(x_t)/dt = noise - action       # ground truth 속도 (t 와 무관)
        noise, t 를 넘기면 그걸 사용 (val: 매번 같은 문제로 채점), 없으면 랜덤 (train)
        '''
        action = (batch["action"] - self.action_mean) / self.action_std   # normalize [B, 16, 2]
        B = action.shape[0]
        if noise is None:
            noise = torch.randn_like(action)                                        # [B, 16, 2]
        if t is None:
            t = torch.distributions.Beta(self.cfg.time_beta_alpha, self.cfg.time_beta_beta).sample((B,)).to(action.device) * 0.999 + 0.001   # [B]  (SmolVLA 와 같은 분포)
        tt = t[:, None, None]                                                       # [B, 1, 1] broadcast 용
        x_t = tt * noise + (1 - tt) * action                                        # [B, 16, 2]
        u_t = noise - action                                                        # [B, 16, 2]
        return x_t, t, u_t

    def suffix_preprocessing(self, x_t, t):
        '''
        x_t: [B, 16, 2]  (데이터의 action 이 아니라 noise 섞인 것!)
        t  : [B]         (데이터에 없음, flow_matching 에서 랜덤 샘플)
        '''
        from utils import sinusoidal_time_emb
        a_emb = self.action_in_proj(x_t)                                            # [B, 16, D]
        t_emb = sinusoidal_time_emb(t, a_emb.shape[-1])[:, None, :].expand_as(a_emb)  # [B, 16, D]
        suffix = self.action_time_mlp(torch.cat([a_emb, t_emb], dim = -1))         # [B, 16, D]
        return suffix

    def forward(self, suffix, memory, pad_masks):
        '''
        suffix: [B, 16, D]
        memory: [B, 95, 512]
        pad_masks: [B, 95] should be 1D please.
        '''
        h = self.expert_decoder(tgt = suffix, 
                    memory = memory,
                    memory_key_padding_mask = ~pad_masks)
        v_t = self.action_out_proj(h)
        return v_t


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

def compute_loss(prefix, expert, batch, cfg, noise=None, t=None):
    prefix_tokens, att_2d_masks, position_ids, pad_masks = prefix.prefix_preprocessing(batch)
    prefix_out = prefix.forward(prefix_tokens, att_2d_masks, position_ids)
    memory = expert.memory_proj(prefix_out.last_hidden_state) # [B, 95, 960] -> [B, 95, 512]  (hidden_states 는 layer별 tuple)
    x_t, t, u_t = expert.flow_matching(batch, noise, t)
    suffix = expert.suffix_preprocessing(x_t, t)
    v_t = expert.forward(suffix, memory, pad_masks)
    valid = (~batch["action_is_pad"])[..., None]                    # [B, 16, 1]
    loss = ((v_t - u_t) ** 2 * valid).sum() / (valid.sum() * cfg.action_dim)
    return loss

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