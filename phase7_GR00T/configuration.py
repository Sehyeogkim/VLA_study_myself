"""
configuration.py — GR00T N1.7 설계도. LIBERO fine-tune.

VLM 은 인코더-디코더가 아니다.
Cosmos-Reason2-2B (Qwen3-VL) 가 사진 + 문장을 읽고 hidden 에서 멈춘다.
action 은 AlternateVLDiT 가 flow matching 으로 푼다.
"""
from dataclasses import dataclass


# @dataclass
# class Config:
#     batch_size: int = 1
#     chunk_size: int = 16
#     action_dim: int = 2
#     device: str = "cuda"
#     num_layers: int = 12
#     num_heads: int = 12
#     num_hidden: int = 768
#     num_embed: int = 768
#     num_embed_kv: int = 768
#     num_embed_kv_kv: int = 768
#     num_embed_kv_kv_kv: int = 768


@dataclass
class Config:
    device: str = "cuda"
    dtype: str = "bfloat16"

    # LIBERO 한 프레임. 2 는 시간이 아니라 카메라 두 대.
    n_cameras: int = 2
    image_size: int = 256          # 3인칭 image, 손목 image2
    image_crop_size: int = 230     # N1.7 processor 가 학습 때 자르는 크기
    state_dim: int = 8             # eef xyz + axis-angle + gripper 손가락 2
    action_dim: int = 7            # Δxyz + Δ회전 + gripper
    fps: int = 10
    # N1.7 은 미래 40칸(4초)을 한 번에 맞춘다. 패딩용 최대 칸은 132.
    action_horizon: int = 40
    chunk_size: int = 40
    max_state_dim: int = 132
    max_action_dim: int = 132

    # VLM 백본. 문장을 다시 생성하는 decoder 는 없다.
    backbone_name: str = "nvidia/Cosmos-Reason2-2B"
    backbone_type: str = "qwen3-vl"
    backbone_hidden: int = 2048
    select_layer: int = 16         # 이 레이어 hidden 을 DiT 에 넘긴다
    vlm_decoder: str | None = None

    # 토크나이저는 백본에 붙어 있다. AutoTokenizer 로 같은 repo 를 연다.
    tokenizer_name: str = "nvidia/Cosmos-Reason2-2B"
    tokenizer_class: str = "Qwen2Tokenizer"
    padding_side: str = "left"
    # 사진은 토크나이저가 아니라 이 processor 가 패치로 자른다.
    image_processor: str = "Qwen2VLImageProcessor"

    # action 쪽. state 도 VLM 토큰이 아니라 DiT 옆 MLP.
    dit_name: str = "AlternateVLDiT"
    dit_layers: int = 32
    dit_heads: int = 32
    dit_head_dim: int = 48
    dit_hidden: int = 1024
    state_encoder: str = "CategorySpecificMLP"
    action_decoder: str = "CategorySpecificMLP"
    num_denoise_steps: int = 4     # 추론 Euler 횟수
    T: float = 1.0
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0

    # 학습. VLM(Cosmos)은 기본으로 얼린다. 맞는 것은 DiT.
    # DiT 가 약 17억이라 Adam 상태까지 하면 A40 48GB 에서 batch 2 가 시작점.
    batch_size: int = 2
    training_steps: int = 10000
    warmup_steps: int = 500
    lr_dit: float = 1e-4
    lr_backbone: float = 1e-5      # train_backbone=True 일 때만
    train_backbone: bool = False
    state_dropout: float = 0.2     # N1.7 이 state 를 가끔 0 으로 비우는 비율
    grad_clip: float = 1.0
    num_workers: int = 4
    log_every: int = 50
    val_every: int = 1000
    val_batches: int = 8
    val_per_task: int = 5          # 태스크마다 맨 끝 데모 몇 개를 val
    ema_decay: float = 0.999
    save_every: int = 5000
    n_action_steps: int = 8        # eval: chunk 40 중 앞에서 몇 개만 실행하고 다시 예측
