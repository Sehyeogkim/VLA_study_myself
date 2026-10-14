"""
utils.py — Full_script.py 에서 쓰는 helper 함수 모음
"""
import torch
import matplotlib.pyplot as plt


# lerobot/policies/common/vla_utils.py 의 make_att_2d_masks 복사 (openpi 원본)
def make_att_2d_masks(pad_masks, att_masks):
    """
    pad_masks: bool [B, L]  True = 진짜 토큰, False = pad
    att_masks: bool [B, L]  1 = 새 블록 시작, 0 = 앞 블록과 한 덩어리
    return   : bool [B, L, L]  [i, j] = query i 가 key j 를 볼 수 있나
    """
    cumsum = torch.cumsum(att_masks, dim=1)                         # [B, L]
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]         # cumsum[j] <= cumsum[i]
    pad_2d_masks = pad_masks[:, None, :] & pad_masks[:, :, None]    # i, j 둘 다 진짜 토큰
    return att_2d_masks & pad_2d_masks


# mask -> png 저장 (흰색 = 볼 수 있음, 검은색 = 못 봄)
def save_mask_png(mask_2d, path, boundaries=None, title=""):
    """
    mask_2d   : bool [L, L]  (batch 하나만 넣기)
    boundaries: 구간 경계 인덱스 list (빨간 선)  e.g. [64, 94]
    """
    plt.figure(figsize=(7, 7))
    plt.imshow(mask_2d.float().cpu(), cmap="gray", vmin=0, vmax=1)
    for b in boundaries or []:
        plt.axvline(b - 0.5, color="red", lw=1)
        plt.axhline(b - 0.5, color="red", lw=1)
    plt.xlabel("j (key)")
    plt.ylabel("i (query)")
    plt.title(f"{title} {tuple(mask_2d.shape)}")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()
    print("saved:", path)


# lerobot/policies/common/vla_utils.py 의 create_sinusoidal_pos_embedding 축소 복사
def sinusoidal_time_emb(t, dim, min_period=4e-3, max_period=4.0):
    """
    t     : [B]  flow matching time (0~1)
    return: [B, dim]  앞 절반 sin, 뒤 절반 cos (주기 min_period ~ max_period)
    """
    import math
    fraction = torch.linspace(0.0, 1.0, dim // 2, device=t.device)
    period = min_period * (max_period / min_period) ** fraction     # [dim/2]
    x = t[:, None] * (2 * math.pi / period)                         # [B, dim/2]
    return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)          # [B, dim]



class BatchStream:
    """
    DataLoader 를 step 단위로 무한히 꺼내 쓰기 위한 책갈피.
    1 epoch 다 읽으면 (StopIteration) 자동으로 다시 섞어서 처음부터.

        stream = BatchStream(loader)
        batch = stream.next()
    """
    def __init__(self, loader):
        self.loader = loader
        self.it = iter(loader)      # 책갈피
        self.epoch = 0

    def next(self):
        try:
            return next(self.it)                 # 다음 batch
        except StopIteration:                    # 책 다 읽음
            self.epoch += 1
            self.it = iter(self.loader)          # 새로 섞어서 처음부터
            return next(self.it)
