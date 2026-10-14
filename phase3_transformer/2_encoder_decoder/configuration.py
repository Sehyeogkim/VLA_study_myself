from dataclasses import dataclass
import torch

@dataclass
class Config:
    d_model: int = 512
    nhead: int = 8              # transformer.py 에서 cfg.nhead 로 사용
    # Attention Is All You Need: N encoder layers = N decoder layers (one shared hyperparameter).
    num_layers: int = 6
    vocab_size: int = 50257     # tokenizer 로드 후 len(tokenizer) 로 덮어씀
    d_ff: int = 2048
    max_seq_len: int = 64
    dropout: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
