'''
Now In here we are going to train the decoder only transformer model.

1. Download the data set (nanogpt) -> Tiny Shakespeare Dataset
2. Tokenize all of them

'''
from pathlib import Path
from configuration import Config
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"

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
            y2 = [2,3,4,5,6] 번째 토큰 from the data

            x2 = [6,7,8,9,10] 번째 토큰 from the data
            y2 = [7,8,9,10,11] 번째 토큰 from the data

            ...

        이런식으로 데이터를 구축한다. 그리고 하나의 pair {x,y}에 대해서 총 L=5개의 학습용 데이터가 생성 되는ㄴ것
        data 내부에는 이런식으로 들어있고, 이제 이게 casula mask로 예측 하게함.
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
    val_loader = DataLoader(val_ds, batch_size=Config.batch_size, shuffle=True, drop_last=True)

    print(f"tokens: total {N:,} | train {len(train_ids):,} | val {len(val_ids):,}")
    print(f"pairs : train {len(train_ds):,} | val {len(val_ds):,} | batches/epoch {len(train_loader):,}")
    return train_loader, val_loader


def train():
    train_loader, val_loader = Dataloader()
    model = DecoderOnlyTransformer()

    # Step1: Get the data
    for step in range(Config.traning_steps):
        x, y = next(iter(train_loader))
        x, input_ids, position_ids, lang_pad = model.preprocessing(x)


        logits = model(x, lang_pad)

        # Step2: Forward pass
        x = x.to(Config.device)
        y = y.to(Config.device)

        # Step3: Backward pass
        loss = loss_fn(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return


if __name__ == "__main__":
    train()
