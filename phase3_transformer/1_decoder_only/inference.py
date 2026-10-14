"""Load ckpt and autoregressively sample text."""
import argparse
from pathlib import Path
import torch
from decoder_only_transformer import DecoderOnlyTransformer

ROOT = Path(__file__).parent
DEFAULT_CKPT = ROOT / "results" / "ckpt.pt"


def device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="Hello How are you doing?")
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    args = ap.parse_args()

    dev = device()
    model = DecoderOnlyTransformer().to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev, weights_only=True))
   
    #아 그러니까 우리가 토큰을 몇개를 더 생성할지를 마라는게 max new tokens임, 그러니가 conterxt window랑 다른 개념임.
    text = model.generate(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print(text)


if __name__ == "__main__":
    main()
