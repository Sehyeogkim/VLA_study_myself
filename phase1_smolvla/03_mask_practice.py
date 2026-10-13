'''
In here we are going to full yunderestnad how does the 2D mask made by the

make_att_2d_masks method please

Input:
    - pad_masks (1D) , [B, L]
    - att_masks (1D) , [B, L]
Output:
    - att_2d_masks(2D) , [B, L, L]


i (Query)-> J  (Key)
- 일단 padding은 그냥 누구도 참고하지 않도록 어차피 2d mask에서 0 으로 찍히는거고 맞재
- Casual mask 같은 경우는 그냥 att_mask = torch.ones(L)

'''

#이거를 정확하게 이해를 진행해보자. 
# 그러니까, 우리가 i -> J 를 참고할때 참고 못하게 하고싶은 경우 이런 구속 조건만 있잖아. 사실
# 하지만 J -> I 는 참고할 수 있고 이런거 이런 구속조건 이거를 consum의 구동조건을 정화갛게 이해해야할듯


# att_maks = [0,0,0,0,1,1,1] 이라면
# Tokens   = [image1~4 and let's say sate1,2,3 tokens]
# cumsum   = [0,0,0,0,1,2,3] 이런식이고,
# att_2d[i,j] = cumsum[j] <= cumsum [i] 이니까
# [1, 1, 1, 1, 0, 0, 0]
# [1, 1, 1, 1, 0, 0, 0]
# [1, 1, 1, 1, 0, 0, 0]
# [1, 1, 1, 1, 0, 0, 0] -> Full yattention for the image 1~4
# [1, 1, 1, 1, 1, 0, 0] -> state1
# [1, 1, 1, 1, 1, 1, 0] -> state2
# [1, 1, 1, 1, 1, 1, 1] -> state3 (Casusal??)
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

# 1D mask -> 2D mask 를 직접 호출해서 확인
# (tokens, pad_masks, att_masks)
cases = {
    # 내 예시: image 4개 = 한 블록, state 3개 = causal
    "my example": (
        ["img1", "img2", "img3", "img4", "s1", "s2", "s3"],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1],
    ),
    # SmolVLA prefix 축소판: image + lang(+pad) 한 블록, state 는 att=1 로 새 블록
    "smolvla prefix": (
        ["img1", "img2", "lang1", "lang2", "pad", "state"],
        [1, 1, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 1],
    ),
    # 전부 0 -> full attention (BERT 처럼 양방향), pad 만 0
    "full (all zeros)": (
        ["t1", "t2", "t3", "t4", "pad"],
        [1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ),
    # SmolVLA suffix: action 전부 att=1 -> causal
    "smolvla suffix": (
        ["a1", "a2", "a3", "a4", "a5"],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
    ),
}

fig, ax = plt.subplots(2, 2, figsize=(10, 10))
for a, (name, (tokens, pad, att)) in zip(ax.flatten(), cases.items()):
    pad_masks = torch.tensor([pad], dtype=torch.bool)   # [B, L]
    att_masks = torch.tensor([att])                     # [B, L]
    att_2d = make_att_2d_masks(pad_masks, att_masks)    # [B, L, L]

    print(f"== {name} ==")
    print("cumsum:", torch.cumsum(att_masks, dim=1)[0].tolist())
    print(att_2d[0].int())

    a.imshow(att_2d[0].float(), cmap="gray", vmin=0, vmax=1)
    a.set_xticks(range(len(tokens)), tokens, rotation=45)
    a.set_yticks(range(len(tokens)), tokens)
    a.set_xlabel("j (key)")
    a.set_ylabel("i (query)")
    a.set_title(f"{name}\natt={att}")

plt.tight_layout()
out = Path(__file__).parent / "mask_practice.png"
plt.savefig(out, dpi=100)
print("saved:", out)
