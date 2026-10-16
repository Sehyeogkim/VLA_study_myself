'''
Generate captions with the trained model (result/best.pt).

1. Load tokenizer + config (train.py 와 똑같이 맞춤)
2. Load model + best.pt
3. Validation 이미지 몇 장 -> greedy decoding (여기서만 argmax)
4. 정답 캡션이랑 나란히 출력
5. KV cache 있는 generate vs 없는 generate -> 토큰 같은지 + 속도 비교
'''
import time
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer

from configuration import Config
from modeling_ver1_KVcache import CaptioningModel
from train import build_image_transform

OUT = Path(__file__).parent / "result"


@torch.no_grad()
def generate(model, pixel_values, cfg, max_new_tokens=30):
    memory = model.encoder(pixel_values)                     # [1, 197, D]  이미지는 한 번만 encode
    ids = torch.tensor([[cfg.bos_token_id]], device=pixel_values.device)
    for _ in range(max_new_tokens):
        logits, _ = model.decoder(ids, memory)               # [1, L, V]  매번 전체 문장 다시 계산
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)   # 마지막 위치의 1등 토큰
        if next_id.item() == cfg.eos_token_id:
            break
        ids = torch.cat([ids, next_id], dim=1)
    return ids[0, 1:].tolist()                               # BOS 빼고 반환


@torch.no_grad()
def generate_cache(model, pixel_values, cfg, max_new_tokens=30):
    memory = model.encoder(pixel_values)                     # [1, 197, D]
    ids = torch.tensor([[cfg.bos_token_id]], device=pixel_values.device)
    logits, cache = model.decoder(ids, memory)               # 첫 step: BOS 넣고 cache 생성
    for _ in range(max_new_tokens):
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)   # [1, 1]
        if next_id.item() == cfg.eos_token_id:
            break
        ids = torch.cat([ids, next_id], dim=1)
        logits, cache = model.decoder(next_id, memory, cache)    # 새 토큰 1 개만 넣음 (나머지는 cache)
    return ids[0, 1:].tolist()


def main(n_images=10):
    # 1. train.py 와 똑같은 tokenizer / config
    cfg = Config()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    cfg.vocab_size = tokenizer.vocab_size
    cfg.pad_token_id = tokenizer.eos_token_id
    cfg.eos_token_id = tokenizer.eos_token_id
    cfg.bos_token_id = tokenizer.bos_token_id or cfg.eos_token_id

    # 2. model + checkpoint
    model = CaptioningModel(cfg).to(cfg.device)
    ckpt = torch.load(OUT / "best.pt", map_location=cfg.device)
    model.load_state_dict(ckpt["model"]) # 가중치를 복사해서 넣는다.
    model.eval() # 이벨 모드 온
    print(f"loaded best.pt (step {ckpt['step']}, val {ckpt['val']:.4f})\n")

    # 3. validation 이미지 -> 캡션 생성
    data = load_dataset("jxie/flickr8k", split="validation")
    transform = build_image_transform(cfg)
    for i in range(n_images):
        row = data[i * 100]                                  # 서로 다른 사진 고르기 위해 100 간격
        pixel_values = transform(row["image"].convert("RGB")).unsqueeze(0).to(cfg.device)   # [1, 3, 224, 224]

        t0 = time.perf_counter(); out = generate(model, pixel_values, cfg); t1 = time.perf_counter()
        out_c = generate_cache(model, pixel_values, cfg);                  t2 = time.perf_counter()
        pred = tokenizer.decode(out)

        # 4. 정답이랑 비교  5. cache 있/없 비교
        print(f"[{i}] pred : {pred}")
        print(f"    cache same: {out == out_c} | no-cache {(t1-t0)*1000:.1f} ms | cache {(t2-t1)*1000:.1f} ms")
        print(f"    gt   : {row['caption_0']}")
        print(f"    gt   : {row['caption_1']}\n")


if __name__ == "__main__":
    main()
