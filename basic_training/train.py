'''
기본적으로 '학습'의 개념에 대해서 정확하게 하고 가야할듯 일단 내 생각은

1. 데이터 1개 스키마 확인
2. 데이터 class제작 -> __init__, __len__, __getitem__
3. Data loader 호출 , train / val split
4. 모델 이미 있다는 전제하에 호출
5. optimizer, loss function 정의
6. train loop 작성
7. 중간에 valiadation 확인
8. 학습 완료 후 모델 저장
'''

from pathlib import Path

from torch.utils.data import DataLoader
from datasets import load_dataset

import numpy as np
import torch

OUT = Path(__file__).parent / "result"


class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(784, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 10),
        )

    def forward(self, x):          # [B, 784]
        return self.net(x)         # [B, 10]


class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        image = row["image"]   # PIL 28x28
        label = row["label"]   # int 0~9

        # PIL -> [784] float 0~1  (MLP 입력)
        image = torch.tensor(np.array(image), dtype=torch.float32) / 255.0
        image = image.view(-1)
        label = torch.tensor(label, dtype=torch.long)

        return {"image": image, "label": label}


def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # 1. Dataset
    data = load_dataset("ylecun/mnist", split="train")
    n = int(len(data) * 0.9)
    train_data = data.select(range(0, n))
    val_data = data.select(range(n, len(data)))

    train_ds = MNISTDataset(train_data)
    val_ds = MNISTDataset(val_data)

    # 2. Data Loader
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    print(f"train {len(train_ds)} | val {len(val_ds)} | device {device}", flush=True)

    # 3. Model / optimizer / loss function
    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    def run_val():
        model.eval()
        tot = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                tot += criterion(model(images), labels).item()
        model.train()
        return tot / len(val_loader)

    # 4. Train loop (step 단위) + 5. Validation
    OUT.mkdir(exist_ok=True)
    best = float("inf")
    it = iter(train_loader)

    v = run_val()
    print(f"step 0 | val {v:.4f}  (학습 전)", flush=True)

    for step in range(1, 5 * len(train_loader) + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()

        if step % 200 == 0:
            print(f"step {step} | loss {loss.item():.4f}", flush=True)

        if step % len(train_loader) == 0:
            v = run_val()
            print(f"step {step} | val {v:.4f}", flush=True)
            if v < best:
                best = v
                torch.save(model.state_dict(), OUT / "best.pt")

    print(f"best val {best:.4f} -> {OUT / 'best.pt'}", flush=True)


if __name__ == "__main__":
    train()
