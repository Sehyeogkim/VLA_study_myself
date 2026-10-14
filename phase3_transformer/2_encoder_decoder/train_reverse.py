"""
train_reverse.py — encoder-decoder 구조 검증용 장난감 과제: 문자열 뒤집기

    src : "hello"            → Encoder (양방향 self-attn) → memory
    dec 입력 : [BOS] o l l e h  → Decoder (causal self-attn + cross-attn → memory)
    정답     : o l l e h [EOS]   (한 칸 민 것 = teacher forcing)

뒤집기는 decoder 가 cross-attention 으로 입력을 "거꾸로" 읽어야만 풀 수 있음
→ 정확도 ~100% 가 나오면 encoder / cross-attn / causal / BOS·EOS / 생성 루프 가 전부 맞다는 증거.

tokenizer: 문자 단위 (a~z + PAD/BOS/EOS = 29 개). GPT-2 tokenizer 안 씀.
모델 크기: encoder 384 / decoder 256 (서로 다르게 → cross-attn 의 W_k, W_v 384→256 도 같이 검증)

실행 (RunPod): python train_reverse.py
"""
import json
import math
import random
import string
import time

import torch
import torch.nn.functional as F

from configuration import OUTPUT_DIR, BlockConfig, Config
from modeling import DecoderTransformer, EncoderTransformer

# ---------- 문자 단위 vocab ----------
PAD, BOS, EOS = 0, 1, 2
CHARS = string.ascii_lowercase                       # a ~ z
STOI = {c: i + 3 for i, c in enumerate(CHARS)}       # a=3 ... z=28
ITOS = {i: c for c, i in STOI.items()}
VOCAB = 3 + len(CHARS)                               # 29

MIN_LEN, MAX_LEN = 5, 20
L = MAX_LEN + 2                                      # BOS/EOS 자리 포함 길이 (22)

# ---------- 학습 설정 ----------
STEPS = 3000
BATCH = 128
LR = 3e-4
WARMUP = 200
LOG_EVERY = 100
VAL_EVERY = 500
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def encode(s):
    return [STOI[c] for c in s]


def decode(ids):
    out = []
    for i in ids:
        if i == EOS:
            break
        if i in ITOS:
            out.append(ITOS[i])
    return "".join(out)


def make_batch(B, rng):
    '''
    src_ids [B, L] : 문자열 + PAD
    dec_in  [B, L] : BOS + 뒤집은 문자열 + PAD
    label   [B, L] : 뒤집은 문자열 + EOS + PAD(-100 으로 무시)
    '''
    src = torch.full((B, L), PAD, dtype=torch.long)
    dec_in = torch.full((B, L), PAD, dtype=torch.long)
    label = torch.full((B, L), -100, dtype=torch.long)
    texts = []
    for b in range(B):
        n = rng.randint(MIN_LEN, MAX_LEN)
        s = "".join(rng.choice(CHARS) for _ in range(n))
        texts.append(s)
        r = encode(s[::-1])
        src[b, :n] = torch.tensor(encode(s))
        dec_in[b, : n + 1] = torch.tensor([BOS] + r)
        label[b, : n + 1] = torch.tensor(r + [EOS])
    src_pad = (src != PAD).long()          # 1 = 진짜 토큰
    dec_pad = (dec_in != PAD).long()
    return src, src_pad, dec_in, dec_pad, label, texts


@torch.no_grad()
def greedy_generate(encoder, decoder, src, src_pad, max_new=L):
    '''BOS 에서 시작 → 매번 마지막 위치 argmax → 붙이기 → 전부 EOS 면 중단'''
    memory = encoder(src, src_pad)                                   # [B, L, D_enc]
    B = src.shape[0]
    ids = torch.full((B, 1), BOS, dtype=torch.long, device=src.device)
    done = torch.zeros(B, dtype=torch.bool, device=src.device)
    for _ in range(max_new):
        logits = decoder(ids, torch.ones_like(ids), memory, src_pad)  # [B, T, V]
        nxt = logits[:, -1, :].argmax(-1)                             # [B]
        nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
        ids = torch.cat([ids, nxt[:, None]], dim=1)
        done = done | (nxt == EOS)
        if done.all():
            break
    return ids[:, 1:]                                                # BOS 제외


def main():
    torch.manual_seed(0)
    cfg = Config(
        encoder=BlockConfig(d_model=384, nhead=8, d_ff=1536, num_layers=3),
        decoder=BlockConfig(d_model=256, nhead=8, d_ff=1024, num_layers=3),
        max_seq_len=64,
        vocab_size=VOCAB,                    # 미리 정해서 → modeling.py 가 GPT-2 tokenizer 안 불러옴
    )
    encoder = EncoderTransformer(cfg).to(DEVICE)
    decoder = DecoderTransformer(cfg).to(DEVICE)
    params = list(encoder.parameters()) + list(decoder.parameters())
    print(f"device {DEVICE} | vocab {VOCAB} | params {sum(p.numel() for p in params):,}")

    opt = torch.optim.AdamW(params, lr=LR)

    def lr_lambda(s):                        # warmup + cosine (phase2 와 같은 모양)
        if s < WARMUP:
            return (s + 1) / WARMUP
        p = (s - WARMUP) / max(1, STEPS - WARMUP)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # 고정 val set (매번 같은 시험지)
    val = make_batch(512, random.Random(123))
    val = tuple(t.to(DEVICE) if torch.is_tensor(t) else t for t in val)

    def evaluate():
        encoder.eval(); decoder.eval()
        src, src_pad, dec_in, dec_pad, label, texts = val
        with torch.no_grad():
            memory = encoder(src, src_pad)
            logits = decoder(dec_in, dec_pad, memory, src_pad)
            vloss = F.cross_entropy(logits.reshape(-1, VOCAB), label.reshape(-1), ignore_index=-100).item()
        out = greedy_generate(encoder, decoder, src, src_pad)
        preds = [decode(o.tolist()) for o in out]
        acc = sum(p == t[::-1] for p, t in zip(preds, texts)) / len(texts)
        encoder.train(); decoder.train()
        return vloss, acc, list(zip(texts[:5], preds[:5]))

    rng = random.Random(0)
    hist = {"train": [], "val": []}
    t0 = time.time()
    for step in range(1, STEPS + 1):
        src, src_pad, dec_in, dec_pad, label, _ = make_batch(BATCH, rng)
        src, src_pad, dec_in, dec_pad, label = [t.to(DEVICE) for t in (src, src_pad, dec_in, dec_pad, label)]

        memory = encoder(src, src_pad)                               # [B, L, 384]
        logits = decoder(dec_in, dec_pad, memory, src_pad)           # [B, L, 29]
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), label.reshape(-1), ignore_index=-100)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()

        if step == 1 or step % LOG_EVERY == 0:
            hist["train"].append((step, loss.item()))
            print(f"step {step:5d} | loss {loss.item():.4f} | lr {opt.param_groups[0]['lr']:.2e} | "
                  f"{step / (time.time() - t0):.1f} it/s", flush=True)
        if step % VAL_EVERY == 0 or step == STEPS:
            vloss, acc, ex = evaluate()
            hist["val"].append((step, vloss, acc))
            print(f"step {step:5d} | val loss {vloss:.4f} | exact-match acc {acc * 100:.1f}%", flush=True)
            for t, p in ex[:3]:
                print(f"    {t:>20s} -> {p:<20s} {'OK' if p == t[::-1] else 'X'}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"encoder": encoder.state_dict(), "decoder": decoder.state_dict()}, OUTPUT_DIR / "reverse.pt")
    json.dump(hist, open(OUTPUT_DIR / "reverse_history.json", "w"))
    print("saved:", OUTPUT_DIR / "reverse.pt")


if __name__ == "__main__":
    main()
