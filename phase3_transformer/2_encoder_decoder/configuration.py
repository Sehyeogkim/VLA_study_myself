from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

# Repo layout (train.py / data later)
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "results"


@dataclass
class BlockConfig:
    """한 쪽 stack (encoder 또는 decoder) 의 크기. TransformerBlock / Attention 은 이것만 보고 만들어짐."""
    d_model: int = 512
    nhead: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    num_layers: int = 6

    def __post_init__(self) -> None:
        if self.d_model % self.nhead != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by nhead ({self.nhead})")


@dataclass
class Config:
    """Hyperparameters for encoder–decoder seq2seq (Attention Is All You Need style)."""

    # --- Model: encoder / decoder 를 따로 (HF 의 vision_config / text_config 와 같은 방식) ---
    # encoder 차원(1024) != decoder 차원(512) -> cross-attn 의 W_k, W_v 가 1024 -> 512 로 맞춤
    encoder: BlockConfig = field(default_factory=lambda: BlockConfig(d_model=1024, d_ff=4096))
    decoder: BlockConfig = field(default_factory=lambda: BlockConfig(d_model=512, d_ff=2048))
    # Positional / sequence limits (embeddings sized by max_seq_len).
    max_seq_len: int = 64
    # Set in EncoderTransformer / DecoderTransformer __init__ via len(tokenizer).
    vocab_size: Optional[int] = None

    # --- Tokenizer (modeling.py) ---
    tokenizer_name: str = "distilgpt2"
    # Padding/truncation length for batching; usually <= max_seq_len.
    tokenizer_max_length: int = 64

    # --- Device ---
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )

    # --- Training (train.py — mirror phase1 decoder-only) ---
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 0.0
    training_steps: int = 2000
    val_every: int = 250
    grad_clip: float = 1.0
    train_split: float = 0.9
    # CrossEntropyLoss: ignore padded target positions (HF pad often 0; set after tokenizer if needed).
    label_ignore_index: int = -100
    seed: int = 42

    # --- Logging / checkpoints ---
    log_every: int = 50
    ckpt_name: str = "best.pt"

    def __post_init__(self) -> None:
        if self.tokenizer_max_length > self.max_seq_len:
            raise ValueError(
                f"tokenizer_max_length ({self.tokenizer_max_length}) must be <= max_seq_len ({self.max_seq_len})"
            )
