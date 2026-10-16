'''
교과서식 학습: epoch 단위.

train.py 는 step + next(it) (phase 0~3 과 같은 방식).
이쪽은 for batch in loader 로 데이터 한 바퀴 = epoch 1 회.
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
        image = row["image"]
        label = row["label"]

        image = torch.tensor(np.array(image), dtype=torch.float32) / 255.0
        image = image.view(-1)
        label = torch.tensor(label, dtype=torch.long)
        return {"image": image, "label": label}


def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    data = load_dataset("ylecun/mnist", split="train")
    n = int(len(data) * 0.9)
    train_ds = MNISTDataset(data.select(range(0, n)))
    val_ds = MNISTDataset(data.select(range(n, len(data))))

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    print(f"train {len(train_ds)} | val {len(val_ds)} | device {device}", flush=True)

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

    OUT.mkdir(exist_ok=True)
    NUM_EPOCHS = 5
    best = float("inf")

    v = run_val()
    print(f"epoch 0 | val {v:.4f}  (학습 전)", flush=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        tot = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            tot += loss.item()

        train_loss = tot / len(train_loader)   # 한 바퀴 평균
        val_loss = run_val()
        print(f"epoch {epoch} | train {train_loss:.4f} | val {val_loss:.4f}", flush=True)

        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), OUT / "best_basic.pt")

    print(f"best val {best:.4f} -> {OUT / 'best_basic.pt'}", flush=True)


if __name__ == "__main__":
    train()
