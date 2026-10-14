from dataclasses import dataclass


@dataclass
class Config:
    d_model: int = 512
    nhead: int = 8
    dim_feedforward: int = 2048
    num_layers: int = 6
    dropout: float = 0.1
    max_seq_len: int = 1024  # model positional limit
    tokenizer_name: str = "distilgpt2"
    tokenizer_max_length: int = 32  # padding length for batching in main / train
    vocab_size: int = None

    # Training parameters
    L: int = 128 # this is the context window.
    batch_size: int = 32
    lr: float = 1e-4
    training_steps: int = 2000
