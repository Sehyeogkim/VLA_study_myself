"""
configuration.py — 모델 설계도 (smolvla: configuration_smolvla.py)

역할: 하이퍼파라미터만 모아두는 dataclass. 학습/추론 코드는 여기 없음.
Target: PushT (state 2D, action 2D, image 96x96)
"""
from dataclasses import dataclass


@dataclass
class MyPolicyConfig:
    # TODO: input/output shape (state_dim, action_dim, image_size)
    # TODO: action chunk (chunk_size, n_action_steps)
    # TODO: model (vision backbone, freeze 여부, hidden_dim, n_layers, n_heads)
    # TODO: flow matching (num_steps, time sampling beta 파라미터)
    # TODO: optimizer / scheduler (lr, weight_decay, warmup_steps)
    pass
