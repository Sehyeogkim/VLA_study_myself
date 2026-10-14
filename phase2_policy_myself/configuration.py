"""
configuration.py — 모델 설계도 (smolvla: configuration_smolvla.py)

역할: 모델 쪽 하이퍼파라미터만 모아두는 dataclass. 학습/추론 코드는 여기 없음.
  - 모델이 바뀌면 같이 바뀌는 값      -> 여기
  - 실험할 때마다 바꾸는 값 (steps, batch_size, num_workers) -> train 스크립트 argparse
Target: PushT (state 2D, action 2D, image 96x96, 10fps)
"""
from dataclasses import dataclass


@dataclass
class MyPolicyConfig:
    # input / output
    state_dim: int = 2              # PushT agent (x, y)
    action_dim: int = 2             # PushT action (x, y)
    fps: int = 10                   # dataset fps (delta_timestamps 계산용)
    chunk_size: int = 16            # action chunk 길이 (10fps -> 1.6초)

    # VLM (freeze)
    vlm_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    vlm_hidden: int = 960           # text_model hidden (= img/lang/state 토큰 차원)
    image_size: int = 512           # SigLIP 입력 크기 (96 -> 512 resize)
    max_lang_len: int = 30          # task 토큰 padding 길이

    # Action expert (B: cross-attention decoder)
    expert_hidden: int = 512        # D
    expert_layers: int = 4
    expert_heads: int = 8

    # flow matching
    num_steps: int = 10             # 추론 Euler step 수
    time_beta_alpha: float = 1.5    # t ~ Beta(1.5, 1) * 0.999 + 0.001 (SmolVLA 와 같음)
    time_beta_beta: float = 1.0

    # optimizer / scheduler preset (LeRobot 처럼 config 에 기본값, CLI 로 덮어쓰기 가능)
    lr: float = 1e-4
    warmup_steps: int = 20
    min_lr_ratio: float = 0.025     # cosine 끝 = lr * 0.025
    grad_clip: float = 1.0
