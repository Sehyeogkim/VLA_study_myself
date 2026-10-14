'''
Now In here we are going to train the decoder only transformer model.

1. Download the data set (nanogpt) -> Tiny Shakespeare Dataset
2. Define the TextDataset class (they have a rule for this make __init__, __len__, __getitem__) -> we need to follow this rule.
3. Make Dataloader instance from the TestDataset.
4. Main train function.
    4.1. Data loading.
    4.2. Model definition (from the decoder_only_transformer.py)
    4.3. Optimizer definition (from the torch.optim)
    4.4. Validation function.
    4.5. Main training loop.
    4.6. Save the model and the loss history.

'''
from torch.nn import functional as F
from pathlib import Path
from configuration import Config
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader
from decoder_only_transformer import DecoderOnlyTransformer
import time, json

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "results"

class TextDataset(torch.utils.data.Dataset):
    '''
    이렇게 정의를 한다음에 인스텐스를 그대로 DAtaloader에게 넘겨주면 알아서 배치 생성함.
    '''
    def __init__(self, ids, L):
        self.ids, self.L = ids, L
    def __len__(self):
        return len(self.ids) - self.L          # 가능한 시작점 개수
    def __getitem__(self, i):
        x = self.ids[i : i + self.L]
        y = self.ids[i + 1 : i + self.L + 1]
        return x, y

def Dataloader():
    # Step1 Load the text data and tokenize it
    tokenizer = AutoTokenizer.from_pretrained(Config.tokenizer_name)
    text = open(DATA_DIR / "input.txt", "r").read()
    ids = torch.tensor(tokenizer(text).input_ids) # [N] 긴 토큰 한 줄

    # Step2 Split the data into train and val
    N = len(ids)
    train_ids = ids[:int(N*0.9)]
    val_ids = ids[int(N*0.9):]

    # Step3 Divide the batch
    '''
        일단 이거 데이터 pair생성 방식과 학습 원리를 이해하고 넘어가자.
        if 5 = L (context window)
            x1 = [1,2,3,4,5] 번째 토큰 from the data
            y1 = [2,3,4,5,6] 번째 토큰 from the data

            x2 = [6,7,8,9,10] 번째 토큰 from the data
            y2 = [7,8,9,10,11] 번째 토큰 from the data

            ...

        이런식으로 데이터를 구축한다. 그리고 하나의 pair {x,y}에 대해서 총 L=5개의 학습용 데이터가 생성 되는 것
        data 내부에는 이런식으로 들어있고, 이제 이게 causal mask로 예측 하게함.
            data1 = input: [1]           -> output: [2]
            data2 = input: [1,2]         -> output: [3]
            data3 = input: [1,2,3]       -> output: [4]
            data4 = input: [1,2,3,4]     -> output: [5]
            data5 = input: [1,2,3,4,5]   -> output: [6]
    '''

    train_ds = TextDataset(train_ids, Config.L)     # i 번째 = (x, y) pair, 요청할 때 잘라줌
    val_ds = TextDataset(val_ids, Config.L)

    # DataLoader: 시작점 섞기 + B 개씩 쌓기 -> x [B, L], y [B, L]
    train_loader = DataLoader(train_ds, batch_size=Config.batch_size, shuffle=True, drop_last=True)
    # val: seed 고정 -> 섞는 순서가 항상 같음 (매번 같은 시험지)
    val_loader = DataLoader(val_ds, batch_size=Config.batch_size, shuffle=True, drop_last=True,
                            generator=torch.Generator().manual_seed(0))

    print(f"tokens: total {N:,} | train {len(train_ids):,} | val {len(val_ids):,}")
    print(f"pairs : train {len(train_ds):,} | val {len(val_ds):,} | batches/epoch {len(train_loader):,}")
    return train_loader, val_loader

def main_train():

    #4.1 Data loading.
    train_loader, val_loader = Dataloader()
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = DecoderOnlyTransformer().to(device)

    #4.2 define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.lr)

    # iterator 는 loop 밖에서 한 번만 (매 step iter() 하면 30만 개를 매번 다시 섞음)
    train_iter = iter(train_loader)

    # val: 전체(약 1,050 batch)는 느리니까 앞에서 batch 20 개(= 640 pair)만 한 번 꺼내서 고정
    from itertools import islice
    val_batches = list(islice(val_loader, 20))
    
    # Average of the loss on the 20 batches for validation.
    def val_loss():
        model.eval()
        with torch.no_grad():
            tot = 0.0
            for vx, vy in val_batches:
                vx, vy = vx.to(device), vy.to(device)
                vl = model(vx, torch.ones_like(vx))
                tot += F.cross_entropy(vl.view(-1, vl.size(-1)), vy.view(-1)).item()
        model.train()
        return tot / len(val_batches)
   
    # Output directory creation.
    OUTPUT_DIR.mkdir(exist_ok=True)
    hist = {"train": [], "val": []}
    t0 = time.time()

    #4.5 Main training loop.
    for step in range(1, Config.training_steps + 1):
        try:
            x, y = next(train_iter)
        except StopIteration:                       # 1 epoch 끝나면 다시 섞어서 처음부터
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        x, y = x.to(device), y.to(device)           # forward 전에 device 로

        #4.6 Forward + loss
        logits = model(x, torch.ones_like(x))                 # [B, L, vocab]
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))  # [B*L, vocab] vs [B*L]

        #4.7 Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 1 or step % 50 == 0:
            hist["train"].append((step, loss.item()))
            print(f"step {step:5d} | loss {loss.item():.4f} | {step / (time.time() - t0):.2f} it/s", flush=True)   # step 1 ≈ ln(50257) = 10.8 이면 정상
        if step % 250 == 0 or step == Config.training_steps:
            v = val_loss(); hist["val"].append((step, v))
            print(f"step {step:5d} | val {v:.4f}", flush=True)

    #4.8 Save the model and the loss history.
    torch.save(model.state_dict(), OUTPUT_DIR / "ckpt.pt")
    json.dump(hist, open(OUTPUT_DIR / "history.json", "w"))
    print("saved:", OUTPUT_DIR / "ckpt.pt")


if __name__ == "__main__":
    main_train()
