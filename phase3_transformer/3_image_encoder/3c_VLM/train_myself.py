'''
Script for the VLM trainign my self.

목표: 데이터셋을 보고 구조화 -> 학습용으로 전환을 이해하기.
'''
from datasets import load_dataset

# data = load_dataset("HuggingFaceM4/the_cauldron", "cocoqa", split="train", streaming = True)

# print("--------------------------------")
# print(data.features)
# print("--------------------------------")

# for sample in data:
#     print(sample["images"][0].size)
#     print(sample["texts"])
#     break

import torch
from modeling import VLM, Config

class VLMDataset(torch.utils.data.Dataset):
    
    def __init__(self, data, data_indices, tokenizer, processor):
        self.data = data
        self.tokenizer = tokenizer
        self.processor = processor
        self.data_indices = data_indices

        texts = data.select_columns(["texts"])["texts"]
        self.pairs = []
        for indic in data_indices:
            for turn_i in range(len(texts[indic])):
                self.pairs.append((indic, turn_i))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        '''
            output schema !!! token not embeddings !!
            {   
                "pixel_values": [3,224,224], (just bf the encoder)
                "input_ids": [user_tokens | assistant_tokens] (L)
                "labels"   : [-100's | answer_tokens] (L)
            }
        '''
        indic, turn_i = self.pairs[idx]
        raw = self.data[indic]

        turn = raw["texts"][turn_i]
        image = raw["images"][0]

        #Image -> pixel values
        pixel_values = self.processor(image, return_tensors="pt")["pixel_values"][0]

        # Tokenizer makes user + assistant -> input_ids.
        input_ids = self.tokenizer.apply_chat_template(
            [
                {"role": "user", "content": turn["user"]},
                {"role": "assistant", "content": turn["assistant"]},
            ],
            tokenize=True, add_generation_prompt=False, return_tensors="pt",
        )["input_ids"][0]                                                       # [L]

        # we need user prompt length.
        prompt_len = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": turn["user"]}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt",
        )["input_ids"].size(1)

        # user prompt -> -100 for the loss calculation.
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        output = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "labels": labels,
        }

        return output



def train_main():

    cfg = Config()
    from transformers import AutoTokenizer, SiglipImageProcessor
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    processor = SiglipImageProcessor.from_pretrained("google/siglip-base-patch16-224")

    data = load_dataset("HuggingFaceM4/the_cauldron", "cocoqa", split="train")
    N = len(data)

    train_data_indices = range(0, int(N * 0.9))
    val_data_indices = range(int(N * 0.9), N)

    train_ds = VLMDataset(data, train_data_indices, tokenizer, processor)
    val_ds = VLMDataset(data, val_data_indices, tokenizer, processor)
    
    #Dataloader
    from torch.utils.data import DataLoader
    def make_collate(pad_token_id):
        def collate(batch):
            '''
            batch -> new batch for the training the model.
            that's the purpose of the collate function.
            '''
            max_token_len = max(data["input_ids"].size(0) for data in batch)
            attention_masks = []
            for data in batch:
                input_ids = data["input_ids"]
                labels = data["labels"]

                #전체 길이가 max_token_len = padding token id 로 먼저 채워넣는다.
                # 기존의 input ids를 채워넣는다.
                new_ids = torch.full((max_token_len,), pad_token_id, dtype=input_ids.dtype)
                new_labels = torch.full((max_token_len,), -100, dtype=labels.dtype)
                new_ids[:input_ids.size(0)] = input_ids
                new_labels[:labels.size(0)] = labels

                data["input_ids"] = new_ids
                data["labels"] = new_labels

                # 진짜 토큰 = 1, pad = 0 (pad 를 모델이 안 보게)
                new_mask = torch.zeros(max_token_len, dtype=torch.long)
                new_mask[:input_ids.size(0)] = 1
                attention_masks.append(new_mask)

            # batch (dict 리스트) -> new_batch (텐서 dict, B 개를 쌓아서 [B, ...]) = 학습 루프가 받는 batch
            new_batch = {
                "pixel_values": torch.stack([data["pixel_values"] for data in batch]),   # [B, 3, 224, 224]
                "input_ids": torch.stack([data["input_ids"] for data in batch]),         # [B, L]
                "labels": torch.stack([data["labels"] for data in batch]),               # [B, L]
                "attention_mask": torch.stack(attention_masks),                          # [B, L]
            }
            return new_batch

        return collate

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=8, collate_fn=make_collate(tokenizer.pad_token_id), drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=True, collate_fn=make_collate(tokenizer.pad_token_id),
                            generator=torch.Generator().manual_seed(0))

    from itertools import islice
    val_batches = list(islice(val_loader, 20))

    #Model
    from transformers import AutoModelForCausalLM
    model = VLM(cfg).to(cfg.device)

    from pathlib import Path
    # Stage 2: Stage 1 에서 학습한 connector 를 불러와서 이어서 시작 (랜덤 connector 로 LLM 을 풀면 LLM 이 망가짐)
    stage1 = torch.load(Path(__file__).parent / "result" / "best.pt", map_location=cfg.device)
    model.connector.load_state_dict(stage1["connector"])
    print(f"loaded Stage 1 connector (step {stage1['step']}, val {stage1['val']:.4f})", flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in params):,}", flush=True)
    #optimizer
    from torch.optim import AdamW
    # optimizer = AdamW(params, lr=cfg.lr)     # Stage 1: connector 하나, lr 1e-3
    # Stage 2: lr 두 그룹 -> LLM 은 이미 잘 학습돼 있어서 아주 작게 (크게 건드리면 언어 능력을 잊어버림)
    optimizer = AdamW([
        {"params": model.connector.parameters(), "lr": 1e-4},
        {"params": model.lm.parameters(), "lr": 2e-5},
    ])
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(optimizer, cfg.warmup_steps, cfg.training_steps)

    
    #이 함수는 그러니까, BAtch에 있는 텐서들을 device로 옮기는 함수이다. 연산 전에 필요하다.
    def to_device(b):
        return {k: v.to(cfg.device) for k, v in b.items()}
    
    def val_loss():
        model.eval()
        total = 0.0
        with torch.no_grad():
            for b in val_batches:
                total += model(**to_device(b)).loss.item()
        model.train()
        return total / len(val_batches)

    # Training loop

    from pathlib import Path
    OUT = Path(__file__).parent / "result"
    OUT.mkdir(exist_ok=True)
    best = float("inf")
    hist = {"train_loss": [], "val_loss": []}

    import time, json
    t0 = time.time()
    it = iter(train_loader)

    for step in range(1, cfg.training_steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)

        loss = model(**to_device(batch)).loss

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
            if v < best:
                best = v
                # VLM 은 직접 만든 nn.Module (save_pretrained 없음) -> 학습하는 connector 만 저장
                # torch.save({"connector": model.connector.state_dict(), "step": step, "val": v}, OUT / "best.pt")   # Stage 1
                # Stage 2: LLM 도 학습했으니 같이 저장 (Stage 1 best.pt 를 덮어쓰지 않게 파일 이름 분리)
                torch.save({"connector": model.connector.state_dict(), "lm": model.lm.state_dict(), "step": step, "val": v},
                           OUT / "stage2_best.pt")
            # json.dump(hist, open(OUT / "history.json", "w"))   # Stage 1
            json.dump(hist, open(OUT / "stage2_history.json", "w"))


if __name__ == "__main__":
    train_main()