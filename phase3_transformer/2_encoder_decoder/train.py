"""
Encoder–decoder training on local parallel.tsv (EN → KO).

Teacher forcing: decoder input = tgt[:, :-1], labels = tgt[:, 1:], pad masked in loss.
"""

from __future__ import annotations

import json
import random
import time
from itertools import islice
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from configuration import DATA_DIR, OUTPUT_DIR, Config
from modeling import DecoderTransformer, EncoderTransformer

PARALLEL_PATH = DATA_DIR / "parallel.tsv"


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
        enc = self.tokenizer(
            src,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        dec = self.tokenizer(
            tgt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        return {
            "src_ids": enc.input_ids.squeeze(0),
            "src_pad": enc.attention_mask.squeeze(0),
            "tgt_ids": dec.input_ids.squeeze(0),
            "tgt_pad": dec.attention_mask.squeeze(0),
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
        tgt_ids: torch.Tensor,
        tgt_pad: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encoder(src_ids, src_pad)
        dec_ids = tgt_ids[:, :-1]
        dec_pad = tgt_pad[:, :-1]
        return self.decoder(dec_ids, dec_pad, memory, src_pad)


def seq2seq_loss(
    logits: torch.Tensor,
    tgt_ids: torch.Tensor,
    tgt_pad: torch.Tensor,
    ignore_index: int,
) -> torch.Tensor:
    labels = tgt_ids[:, 1:].contiguous()
    pad_mask = tgt_pad[:, 1:].contiguous()
    labels = labels.masked_fill(pad_mask == 0, ignore_index)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=ignore_index,
    )


def build_dataloaders(cfg: Config, tokenizer):
    pairs = load_parallel_pairs(PARALLEL_PATH)
    rng = random.Random(cfg.seed)
    rng.shuffle(pairs)
    n_train = int(len(pairs) * cfg.train_split)
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:]

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
        f"| train batches/epoch {len(train_loader):,}"
    )
    return train_loader, val_loader


def main_train() -> None:
    cfg = Config()
    device = torch.device(cfg.device)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    train_loader, val_loader = build_dataloaders(cfg, tokenizer)
    model = Seq2SeqModel(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    train_iter = iter(train_loader)
    val_batches = list(islice(val_loader, min(20, len(val_loader))))

    def run_val() -> float:
        model.eval()
        total = 0.0
        n = 0
        with torch.no_grad():
            for batch in val_batches:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(
                    batch["src_ids"],
                    batch["src_pad"],
                    batch["tgt_ids"],
                    batch["tgt_pad"],
                )
                total += seq2seq_loss(
                    logits, batch["tgt_ids"], batch["tgt_pad"], cfg.label_ignore_index
                ).item()
                n += 1
        model.train()
        return total / max(n, 1)

    hist: dict[str, list] = {"train": [], "val": []}
    best_val = float("inf")
    t0 = time.time()

    for step in range(1, cfg.training_steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(
            batch["src_ids"],
            batch["src_pad"],
            batch["tgt_ids"],
            batch["tgt_pad"],
        )
        loss = seq2seq_loss(
            logits, batch["tgt_ids"], batch["tgt_pad"], cfg.label_ignore_index
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        if step == 1 or step % cfg.log_every == 0:
            hist["train"].append((step, loss.item()))
            elapsed = time.time() - t0
            print(
                f"step {step:5d} | loss {loss.item():.4f} | {step / max(elapsed, 1e-6):.2f} it/s",
                flush=True,
            )

        if step % cfg.val_every == 0 or step == cfg.training_steps:
            v = run_val()
            hist["val"].append((step, v))
            print(f"step {step:5d} | val {v:.4f}", flush=True)
            if v < best_val:
                best_val = v
                ckpt = {
                    "encoder": model.encoder.state_dict(),
                    "decoder": model.decoder.state_dict(),
                    "cfg": cfg,
                    "step": step,
                    "val_loss": v,
                }
                torch.save(ckpt, OUTPUT_DIR / cfg.ckpt_name)

    torch.save(
        {
            "encoder": model.encoder.state_dict(),
            "decoder": model.decoder.state_dict(),
            "cfg": cfg,
            "step": cfg.training_steps,
        },
        OUTPUT_DIR / "last.pt",
    )
    with (OUTPUT_DIR / "history.json").open("w") as f:
        json.dump(hist, f)
    print("saved:", OUTPUT_DIR / cfg.ckpt_name, OUTPUT_DIR / "last.pt")


if __name__ == "__main__":
    main_train()
