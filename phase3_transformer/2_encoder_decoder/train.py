"""
Encoder–decoder training on local parallel.tsv (EN → KO).

흐름:
  1. 데이터     data/parallel.tsv (영어 \\t 한국어) -> 섞어서 train / val
  2. tokenizer  train 문장으로 BPE 를 직접 학습 (영어+한국어 공용, 8000 개, [PAD]=0 [BOS]=1 [EOS]=2)
                (GPT-2 tokenizer 는 한글을 byte 로 잘게 쪼개서 문장이 잘리고, BOS/PAD 가 없어서 안 씀)
  3. Dataset    src -> src_ids          (+PAD)
                tgt -> dec_in = [BOS] + tgt   (decoder 입력)
                       label  = tgt + [EOS]   (정답, 한 칸 민 것 = teacher forcing)
  4. 모델       memory = Encoder(src) ; logits = Decoder(dec_in, memory)
  5. loss       CrossEntropy(logits, label), PAD 자리 = -100 (무시)  ← EOS 는 무시 안 함 -> 끝내는 법도 배움
  6. 평가       val loss + greedy 번역 몇 개 (BOS 에서 시작, EOS 나오면 멈춤)

실행 (RunPod): python train.py
"""

from __future__ import annotations

import json
import math
import random
import time
from itertools import islice
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from configuration import DATA_DIR, OUTPUT_DIR, BlockConfig, Config
from modeling import DecoderTransformer, EncoderTransformer

PARALLEL_PATH = DATA_DIR / "parallel.tsv"
PAD, BOS, EOS = 0, 1, 2                      # BPE 학습할 때 special token 순서대로 0, 1, 2


def load_parallel_pairs(path: Path) -> list[tuple[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise FileNotFoundError(f"Empty or missing data: {path}")
    start = 1 if lines[0].lower().startswith("src") else 0
    pairs: list[tuple[str, str]] = []
    for line in lines[start:]:
        if not line.strip():
            continue
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        src, tgt = parts[0].strip(), parts[1].strip()
        if src and tgt:
            pairs.append((src, tgt))
    if not pairs:
        raise ValueError(f"No valid pairs in {path}")
    return pairs


def train_bpe_tokenizer(texts: list[str], vocab_size: int):
    '''영어+한국어 문장으로 byte-level BPE 를 직접 학습 (몇 초). special: [PAD]=0 [BOS]=1 [EOS]=2 [UNK]=3'''
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(texts, trainer)
    assert tok.token_to_id("[PAD]") == PAD and tok.token_to_id("[BOS]") == BOS and tok.token_to_id("[EOS]") == EOS
    return tok


class ParallelTranslationDataset(Dataset):
    def __init__(
        self,
        pairs: list[tuple[str, str]],
        tokenizer,
        max_length: int,
    ):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        src, tgt = self.pairs[idx]
        L = self.max_length
        s = self.tokenizer.encode(src).ids[:L]
        t = self.tokenizer.encode(tgt).ids[: L - 1]           # BOS / EOS 자리 1 개 남김

        src_ids = torch.full((L,), PAD, dtype=torch.long)
        dec_in = torch.full((L,), PAD, dtype=torch.long)
        label = torch.full((L,), -100, dtype=torch.long)       # -100 = loss 에서 무시
        src_ids[: len(s)] = torch.tensor(s)
        dec_in[: len(t) + 1] = torch.tensor([BOS] + t)         # decoder 입력:  [BOS] 안 녕
        label[: len(t) + 1] = torch.tensor(t + [EOS])          # 정답 (한 칸 밀림): 안 녕 [EOS]
        return {
            "src_ids": src_ids,
            "src_pad": (src_ids != PAD).long(),
            "dec_in": dec_in,
            "dec_pad": (dec_in != PAD).long(),
            "label": label,
        }


class Seq2SeqModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.encoder = EncoderTransformer(cfg)
        self.decoder = DecoderTransformer(cfg)

    def forward(
        self,
        src_ids: torch.Tensor,
        src_pad: torch.Tensor,
        dec_in: torch.Tensor,
        dec_pad: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encoder(src_ids, src_pad)                     # [B, L_src, D_enc]
        return self.decoder(dec_in, dec_pad, memory, src_pad)      # [B, L_tgt, vocab]

    @torch.no_grad()
    def greedy_translate(self, src_ids, src_pad, max_new: int):
        '''BOS 에서 시작 -> 매번 마지막 위치 argmax 를 붙임 -> 전부 EOS 면 중단'''
        memory = self.encoder(src_ids, src_pad)                     # encoder 는 1 번만
        B = src_ids.shape[0]
        ids = torch.full((B, 1), BOS, dtype=torch.long, device=src_ids.device)
        done = torch.zeros(B, dtype=torch.bool, device=src_ids.device)
        for _ in range(max_new):
            logits = self.decoder(ids, torch.ones_like(ids), memory, src_pad)
            nxt = logits[:, -1, :].argmax(-1)                       # [B]
            nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
            ids = torch.cat([ids, nxt[:, None]], dim=1)
            done = done | (nxt == EOS)
            if done.all():
                break
        return ids[:, 1:]                                           # BOS 제외


def seq2seq_loss(logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    # label 은 Dataset 에서 이미 한 칸 밀려 있고 PAD 자리는 -100
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), label.reshape(-1), ignore_index=-100)


def ids_to_text(tokenizer, ids) -> str:
    ids = list(ids)
    if EOS in ids:
        ids = ids[: ids.index(EOS)]
    return tokenizer.decode([i for i in ids if i != PAD])


def build_dataloaders(cfg: Config):
    pairs = load_parallel_pairs(PARALLEL_PATH)
    rng = random.Random(cfg.seed)
    rng.shuffle(pairs)
    n_train = int(len(pairs) * cfg.train_split)
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:]

    # tokenizer 는 train 문장으로만 학습 (val 정보가 새지 않게)
    tokenizer = train_bpe_tokenizer([t for p in train_pairs for t in p], cfg.bpe_vocab_size)

    train_ds = ParallelTranslationDataset(train_pairs, tokenizer, cfg.tokenizer_max_length)
    val_ds = ParallelTranslationDataset(val_pairs, tokenizer, cfg.tokenizer_max_length)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
    )
    print(
        f"pairs: total {len(pairs):,} | train {len(train_pairs):,} | val {len(val_pairs):,} "
        f"| train batches/epoch {len(train_loader):,} | vocab {tokenizer.get_vocab_size():,}"
    )
    return train_loader, val_loader, val_pairs, tokenizer


def main_train() -> None:
    cfg = Config(
        # 데이터(수만 쌍)에 맞게 작게. encoder != decoder 차원 -> cross-attn W_k, W_v 384 -> 256
        encoder=BlockConfig(d_model=384, nhead=8, d_ff=1536, num_layers=4),
        decoder=BlockConfig(d_model=256, nhead=8, d_ff=1024, num_layers=4),
        batch_size=128,
        lr=5e-4,
        training_steps=10000,
        val_every=500,
        log_every=100,
    )
    device = torch.device(cfg.device)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)

    train_loader, val_loader, val_pairs, tokenizer = build_dataloaders(cfg)
    tokenizer.save(str(OUTPUT_DIR / "tokenizer.json"))
    cfg.vocab_size = tokenizer.get_vocab_size()          # 미리 정해두면 modeling.py 가 GPT-2 tokenizer 안 씀

    model = Seq2SeqModel(cfg).to(device)
    print(f"device {device} | params {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    def lr_lambda(s):                                     # warmup + cosine (phase2 와 같은 모양)
        if s < cfg.warmup_steps:
            return (s + 1) / cfg.warmup_steps
        p = (s - cfg.warmup_steps) / max(1, cfg.training_steps - cfg.warmup_steps)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * p))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    train_iter = iter(train_loader)
    val_batches = list(islice(val_loader, min(20, len(val_loader))))

    def run_val() -> float:
        model.eval()
        total = 0.0
        n = 0
        with torch.no_grad():
            for batch in val_batches:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(batch["src_ids"], batch["src_pad"], batch["dec_in"], batch["dec_pad"])
                total += seq2seq_loss(logits, batch["label"]).item()
                n += 1
        model.train()
        return total / max(n, 1)

    def show_translations(k: int = 4) -> list[dict]:
        '''val 앞 k 문장을 greedy 번역해서 출력'''
        model.eval()
        b = {key: v[:k].to(device) for key, v in val_batches[0].items()}
        out = model.greedy_translate(b["src_ids"], b["src_pad"], max_new=cfg.tokenizer_max_length)
        model.train()
        rows = []
        for i in range(k):
            src, ref = val_pairs[i]
            pred = ids_to_text(tokenizer, out[i].tolist())
            rows.append({"src": src, "ref": ref, "pred": pred})
            print(f"    EN  : {src}\n    REF : {ref}\n    PRED: {pred}")
        return rows

    hist: dict[str, list] = {"train": [], "val": [], "samples": []}
    best_val = float("inf")
    t0 = time.time()

    for step in range(1, cfg.training_steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["src_ids"], batch["src_pad"], batch["dec_in"], batch["dec_pad"])
        loss = seq2seq_loss(logits, batch["label"])

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        if step == 1 or step % cfg.log_every == 0:
            hist["train"].append((step, loss.item()))
            elapsed = time.time() - t0
            print(
                f"step {step:5d} | loss {loss.item():.4f} | lr {optimizer.param_groups[0]['lr']:.2e} | "
                f"{step / max(elapsed, 1e-6):.2f} it/s",
                flush=True,
            )

        if step % cfg.val_every == 0 or step == cfg.training_steps:
            v = run_val()
            hist["val"].append((step, v))
            tag = ""
            if v < best_val:
                best_val = v
                torch.save(
                    {
                        "encoder": model.encoder.state_dict(),
                        "decoder": model.decoder.state_dict(),
                        "cfg": cfg,
                        "step": step,
                        "val_loss": v,
                    },
                    OUTPUT_DIR / cfg.ckpt_name,
                )
                tag = "  <- best (saved)"
            print(f"step {step:5d} | val {v:.4f}{tag}", flush=True)
            hist["samples"].append((step, show_translations()))

    with (OUTPUT_DIR / "history.json").open("w") as f:
        json.dump(hist, f, ensure_ascii=False)
    print("saved:", OUTPUT_DIR / cfg.ckpt_name, OUTPUT_DIR / "history.json", OUTPUT_DIR / "tokenizer.json")


if __name__ == "__main__":
    main_train()
