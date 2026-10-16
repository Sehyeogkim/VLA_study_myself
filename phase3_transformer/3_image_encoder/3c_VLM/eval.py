'''
Generate answers with the trained connector (result/best.pt).

1. Load tokenizer, processor, VLM (SigLIP + SmolLM2 pretrained) + 학습한 connector
2. val 이미지 (train 에서 안 본 뒤쪽 10%) -> 질문 -> model.generate (KV cache)
3. 정답이랑 나란히 출력 + 정답률 (exact match)
'''
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, SiglipImageProcessor

from modeling import VLM, Config

OUT = Path(__file__).parent / "result"


def main(ckpt_name="best.pt", n_print=15, n_acc=300):
    cfg = Config()
    tokenizer = AutoTokenizer.from_pretrained(cfg.lm_name)
    processor = SiglipImageProcessor.from_pretrained(cfg.vision_name)

    # 1. model + 학습한 connector 만 덮어쓰기 (SigLIP, LLM 은 pretrained 그대로)
    model = VLM(cfg).to(cfg.device)
    ckpt = torch.load(OUT / ckpt_name, map_location=cfg.device)
    model.connector.load_state_dict(ckpt["connector"])
    if "lm" in ckpt:                                                        # Stage 2: 학습한 LLM 도 덮어쓰기
        model.lm.load_state_dict(ckpt["lm"])
    model.eval()
    print(f"loaded {ckpt_name} (step {ckpt['step']}, val {ckpt['val']:.4f})\n")

    # 2. val 이미지 = train_myself.py 와 같은 split (뒤쪽 10%)
    data = load_dataset("HuggingFaceM4/the_cauldron", "cocoqa", split="train")
    N = len(data)
    val_indices = range(int(N * 0.9), N)

    correct = 0
    for k in range(n_acc):
        row = data[val_indices[k * 10]]                                     # 서로 다른 이미지
        turn = row["texts"][0]
        pixel_values = processor(row["images"][0], return_tensors="pt")["pixel_values"].to(cfg.device)
        gen_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": turn["user"]}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt",
        )["input_ids"].to(cfg.device)                                       # assistant\n 까지 (여기서부터 생성)

        pred = tokenizer.decode(model.generate(pixel_values, gen_ids, tokenizer.eos_token_id)).strip()
        gt = turn["assistant"].strip()
        correct += pred.lower() == gt.lower()

        # 3. 정답이랑 비교
        if k < n_print:
            mark = "O" if pred.lower() == gt.lower() else "X"
            print(f"[{k:2d}] {mark} Q: {turn['user'].splitlines()[0]}")
            print(f"        pred: {pred!r} | gt: {gt!r}")

    print(f"\nexact match: {correct}/{n_acc} = {correct / n_acc:.1%}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "best.pt")              # python eval.py stage2_best.pt
