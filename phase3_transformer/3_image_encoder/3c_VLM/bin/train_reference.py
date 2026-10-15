'''
Stage 1 training: connector 만 학습 (SigLIP, SmolLM2 는 얼림)

1. VQADataset class    -> __len__, __getitem__ (질문 1 개 = 샘플 1 개 로 펼침)
2. collate (padding)   -> 길이 다른 샘플들을 오른쪽 pad 로 맞춰서 배치
3. DataLoader          -> train / val (이미지 단위로 나눔)
4. Train loop          -> loss (답변에만), val, 샘플 생성, best 저장
'''
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, SiglipImageProcessor, get_cosine_schedule_with_warmup

from modeling import VLM, Config

OUT = Path(__file__).parent / "result"


# =========================================================
# 1. Dataset
# =========================================================
class VQADataset(torch.utils.data.Dataset):
    '''
    cocoqa 한 row = 이미지 1 장 + 질문/답 여러 턴
    -> (이미지 번호, 턴 번호) 쌍으로 펼쳐서 "질문 1 개 = 샘플 1 개"
    '''
    def __init__(self, data, img_indices, tokenizer, processor, max_len):
        self.data, self.tokenizer, self.processor, self.max_len = data, tokenizer, processor, max_len

        # 턴 개수만 알면 되니까 texts 컬럼만 읽음 (이미지 디코딩 X -> 빠름)
        n_turns = [len(t) for t in data.select_columns(["texts"])["texts"]]
        self.pairs = [(i, t) for i in img_indices for t in range(n_turns[i])]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_i, turn_i = self.pairs[idx]
        row = self.data[img_i]
        turn = row["texts"][turn_i]

        # 이미지 -> SigLIP processor (resize 224, normalize 0.5/0.5)
        pixel_values = self.processor(images=row["images"][0].convert("RGB"), return_tensors="pt")["pixel_values"][0]   # [3, 224, 224]

        # 텍스트 -> chat template (modeling.py main() 2, 3 번과 같음)
        messages = [{"role": "user", "content": turn["user"]}, {"role": "assistant", "content": turn["assistant"]}]
        input_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt")["input_ids"][0]
        prompt_len = self.tokenizer.apply_chat_template(
            messages[:1], tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )["input_ids"].size(1)

        labels = input_ids.clone()
        labels[:prompt_len] = -100                                         # 답변에만 loss
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids[: self.max_len],                         # [L] (길이 제각각)
            "labels": labels[: self.max_len],
        }


# =========================================================
# 2. collate: 오른쪽 padding
#    pad = eos = <|im_end|>(2) -> attention_mask 0, labels -100 로 구분
# =========================================================
def make_collate(pad_id):
    def collate(batch):
        L = max(b["input_ids"].size(0) for b in batch)
        B = len(batch)
        input_ids = torch.full((B, L), pad_id, dtype=torch.long)
        labels = torch.full((B, L), -100, dtype=torch.long)
        attention_mask = torch.zeros((B, L), dtype=torch.long)
        for i, b in enumerate(batch):
            n = b["input_ids"].size(0)
            input_ids[i, :n] = b["input_ids"]
            labels[i, :n] = b["labels"]
            attention_mask[i, :n] = 1
        return {
            "pixel_values": torch.stack([b["pixel_values"] for b in batch]),  # [B, 3, 224, 224]
            "input_ids": input_ids,                                            # [B, L]
            "labels": labels,                                                  # [B, L]
            "attention_mask": attention_mask,                                  # [B, L]
        }
    return collate


def main():
    cfg = Config()
    tokenizer = AutoTokenizer.from_pretrained(cfg.lm_name)
    processor = SiglipImageProcessor.from_pretrained(cfg.vision_name)

    # =========================================================
    # 3. DataLoader: 이미지 단위로 train / val 나눔 (같은 이미지의 다른 질문이 섞이지 않게)
    # =========================================================
    data = load_dataset("HuggingFaceM4/the_cauldron", "cocoqa", split="train")
    N = len(data)
    train_ds = VQADataset(data, range(0, N - cfg.val_images), tokenizer, processor, cfg.max_text_len)
    val_ds = VQADataset(data, range(N - cfg.val_images, N), tokenizer, processor, cfg.max_text_len)
    print(f"images {N:,} | train pairs {len(train_ds):,} | val pairs {len(val_ds):,}", flush=True)

    collate = make_collate(tokenizer.pad_token_id)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=8, collate_fn=collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate,
                            generator=torch.Generator().manual_seed(0))

    from itertools import islice
    val_batches = list(islice(val_loader, 20))                              # 고정 시험지

    # =========================================================
    # 4. Model / optimizer / scheduler
    # =========================================================
    model = VLM(cfg).to(cfg.device)
    params = [p for p in model.parameters() if p.requires_grad]             # = connector 만
    print(f"trainable params: {sum(p.numel() for p in params):,}", flush=True)
    optimizer = torch.optim.AdamW(params, lr=cfg.lr)
    scheduler = get_cosine_schedule_with_warmup(optimizer, cfg.warmup_steps, cfg.training_steps)

    def to_dev(b):
        return {k: v.to(cfg.device) for k, v in b.items()}

    def val_loss():
        model.eval()
        tot = 0.0
        with torch.no_grad():
            for b in val_batches:
                tot += model(**to_dev(b)).loss.item()
        model.train()
        return tot / len(val_batches)

    def sample_answers(n=3):
        '''val 이미지 몇 장으로 실제 생성 (KV cache)'''
        for k in range(n):
            img_i, turn_i = val_ds.pairs[k * 50]
            row = data[img_i]
            turn = row["texts"][turn_i]
            pv = processor(images=row["images"][0].convert("RGB"), return_tensors="pt")["pixel_values"].to(cfg.device)
            gen_ids = tokenizer.apply_chat_template([{"role": "user", "content": turn["user"]}], tokenize=True,
                                                    add_generation_prompt=True, return_tensors="pt")["input_ids"].to(cfg.device)
            ans = tokenizer.decode(model.generate(pv, gen_ids, tokenizer.eos_token_id))
            print(f"    Q: {turn['user'].splitlines()[0]} | pred: {ans!r} | gt: {turn['assistant']!r}", flush=True)

    # =========================================================
    # 5. Train loop (step 단위)
    # =========================================================
    OUT.mkdir(exist_ok=True)
    hist = {"train_loss": [], "val_loss": []}
    best = float("inf")
    t0 = time.time()
    it = iter(train_loader)

    v = val_loss(); hist["val_loss"].append((0, v))
    print(f"step 0 | val {v:.4f}  (학습 전)", flush=True)
    sample_answers()

    for step in range(1, cfg.training_steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)

        loss = model(**to_dev(batch)).loss                                  # HF 가 한 칸 밀기 + -100 무시

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()

        if step % 50 == 0:
            hist["train_loss"].append((step, loss.item()))
            print(f"step {step} | loss {loss.item():.4f} | lr {scheduler.get_last_lr()[0]:.2e} | {step / (time.time() - t0):.2f} it/s", flush=True)
        if step % 250 == 0 or step == cfg.training_steps:
            v = val_loss(); hist["val_loss"].append((step, v))
            print(f"step {step} | val {v:.4f}", flush=True)
            sample_answers()
            if v < best:                                                    # connector 만 저장 (작음)
                best = v
                torch.save({"connector": model.connector.state_dict(), "step": step, "val": v}, OUT / "best.pt")
            json.dump(hist, open(OUT / "history.json", "w"))


if __name__ == "__main__":
    main()
