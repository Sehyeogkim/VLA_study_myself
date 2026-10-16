'''
So far we have built the encoder and decoder.
and now we need to train the model.

so overall

1. define the CaptionDataset class
    __len__
    __getitem__ -> pick one of the data from the dataset
'''
from __future__ import annotations

import random

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from configuration import Config

from modeling_ver1 import CaptioningModel

class CaptionDataset(torch.utils.data.Dataset):
    
    def __init__(
        self,
        hf_split: str,
        tokenizer,
        transform: transforms.Compose,
        max_len: int,
    ) -> None:

        DATASET_NAME = "jxie/flickr8k"
        self.data = load_dataset(DATASET_NAME, split=hf_split)
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.pad_id = tokenizer.pad_token_id
        self.eos_id = tokenizer.eos_token_id
        self.bos_id = tokenizer.bos_token_id or tokenizer.eos_token_id

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        
        row = self.data[idx]

        pixel_values = self.transform((row["image"]).convert("RGB")) # black -> RGB.

        k = random.randint(0, 4)
        caption = row[f"caption_{k}"]

        L = self.max_len
        caption_token_ids = self.tokenizer.encode(caption, add_special_tokens=False)[: L - 1]
        n = len(caption_token_ids)

        #pad 먼저 넣고, 그리고 이제 나머지는 bos + caption_token으로 채워넣는다. 
        dec_in = torch.full((L,), self.pad_id, dtype=torch.long)
        dec_in[: n + 1] = torch.tensor([self.bos_id] + caption_token_ids, dtype=torch.long)

        #labels는 caption_token_ids + eos_id로 채워넣는다. 
        labels = torch.full((L,), -100, dtype=torch.long) # -100 -> ignore token not to calcualte loss.
        labels[: n + 1] = torch.tensor(caption_token_ids + [self.eos_id], dtype=torch.long)
        
        #Padding mask.
        dec_attention_mask = (dec_in != self.pad_id).long()

        return {
            "pixel_values": pixel_values,
            "dec_in": dec_in,
            "labels": labels,
            "dec_attention_mask": dec_attention_mask,
        }


def build_image_transform(cfg: Config, *, normalize: bool = True) -> transforms.Compose:
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    steps: list = [
        transforms.Resize((cfg.H, cfg.W)),
        transforms.ToTensor(),
    ]
    if normalize:
        steps.append(transforms.Normalize(imagenet_mean, imagenet_std))
    return transforms.Compose(steps)


def caption_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
    )



def train() -> None:
    
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
    cfg = Config()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Read the rule of our tokenizer that we gonna import.
    cfg.vocab_size = tokenizer.vocab_size
    cfg.pad_token_id = tokenizer.pad_token_id
    cfg.eos_token_id = tokenizer.eos_token_id
    cfg.bos_token_id = tokenizer.bos_token_id or cfg.eos_token_id

    #Now define the dataset class.
    train_ds = CaptionDataset("train", tokenizer, build_image_transform(cfg), max_len=cfg.max_seq_len)
    val_ds = CaptionDataset("validation",tokenizer, build_image_transform(cfg), max_len=cfg.max_seq_len)

    #PUt the instance on the data loader. 
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers = 8)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    from itertools import islice
    val_batches = list(islice(val_loader, 20))
    
    #Define the model.
    model = CaptioningModel(cfg).to(cfg.device)

    #Define the optimizer.  
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    #Define the scheduler. warmup (0 -> lr) 후 cosine 으로 0 까지 내려감
    STEPS = cfg.num_epochs * len(train_loader)          # 20 × 375 = 7,500
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=STEPS)
    
    #Define the validation loss function. ( In order to check wether the training is going well or not.)
    def val_loss():
        model.eval()
        tot = 0.0
        with torch.no_grad():
            for b in val_batches:
                logits = model(b["pixel_values"].to(cfg.device), b["dec_in"].to(cfg.device))
                tot += caption_loss(logits, b["labels"].to(cfg.device)).item()
        model.train()
        return tot / len(val_batches)


    import json, time
    from pathlib import Path
    OUT = Path(__file__).parent / "result"
    OUT.mkdir(exist_ok=True)
    hist = {"train_loss": [], "val_loss": []}
    best = float("inf"); step = 0
    t0 = time.time()

    #########################################################
    #Define the training loop.
    for epoch in range(cfg.num_epochs):
        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(cfg.device)
            dec_in = batch["dec_in"].to(cfg.device)
            labels = batch["labels"].to(cfg.device)
            dec_attention_mask = batch["dec_attention_mask"].to(cfg.device)

            logits = model(pixel_values, dec_in)
            loss = caption_loss(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            
            ##THis is called 'Grad Clip"
            #정확하게는 아주 큰 gradient가 나오면 같은 방향을 유지한채 길이만 줄어들게 한다.
            # 즉 급격한 변화를 억제한다. 
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()





            #########################################################
            #Debugging stuffs
            step += 1
            if step % 50 == 0:
                hist["train_loss"].append((step, loss.item()))
                print(f"ep {epoch} step {step} | loss {loss.item():.4f} | lr {scheduler.get_last_lr()[0]:.2e} | {step/(time.time()-t0):.2f} it/s", flush=True)
            if step % 250 == 0:
                v = val_loss(); hist["val_loss"].append((step, v))
                print(f"step {step} | val {v:.4f}", flush=True)
                if v < best:                       # best일 때만 저장 (overfit 대비)
                    best = v
                    torch.save({"model": model.state_dict(), "step": step, "val": v}, OUT / "best.pt")
                json.dump(hist, open(OUT / "history.json", "w"))


if __name__ == "__main__":
    train()
