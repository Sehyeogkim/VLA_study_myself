from dataclasses import dataclass, field

import torch
@dataclass
class BlockConfig:
    """One transformer stack (vision encoder or caption decoder)."""

    d_model: int = 1024
    nhead: int = 8
    d_ff: int = 2048
    num_layers: int = 6
    # GQA: K/V head count; None -> same as nhead (full MHA)
    num_kv_heads: int | None = None

    def __post_init__(self) -> None:
        if self.d_model % self.nhead != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by nhead ({self.nhead})"
            )
        if self.num_kv_heads is None:
            self.num_kv_heads = self.nhead
        if self.nhead % self.num_kv_heads != 0:
            raise ValueError(
                f"nhead ({self.nhead}) must be divisible by num_kv_heads ({self.num_kv_heads})"
            )


@dataclass
class Config:
    encoder: BlockConfig = field(
        default_factory=lambda: BlockConfig(d_model=512, d_ff=2048, num_layers=4)
    )
    decoder: BlockConfig = field(
        default_factory=lambda: BlockConfig(d_model=512, d_ff=2048, num_layers=4)
    )

    # Image stem (resize/crop in DataLoader should match H, W)
    H: int = 224
    W: int = 224
    P: int = 16
    C: int = 3
    use_cls: bool = True

    # Text / lm_head — defaults for modeling.py debug; train.py overwrites via sync_config_with_tokenizer()
    vocab_size: int = 256
    max_seq_len: int = 32
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    #For training
    batch_size: int = 16
    lr: float = 1e-4
    num_epochs: int = 20
    warmup_steps: int = 200
