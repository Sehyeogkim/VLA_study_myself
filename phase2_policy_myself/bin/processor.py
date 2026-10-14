"""
processor.py — 전처리/후처리 (smolvla: processor_smolvla.py)

pre : raw obs -> normalize (state, image) -> device -> model 입력
post: model 출력 -> unnormalize (action) -> env 에 넣을 action

※ phase1 교훈: dataset stats 를 정확히 넣어야 함 (안 넣으면 loss 4263)
"""
import torch


class Normalizer:
    def __init__(self, stats: dict):
        # TODO: key 별 mean/std (or min/max) 저장
        pass

    def __call__(self, batch: dict) -> dict:
        # TODO: normalize
        pass


class Unnormalizer:
    def __init__(self, stats: dict):
        # TODO
        pass

    def __call__(self, action: torch.Tensor) -> torch.Tensor:
        # TODO: unnormalize
        pass


def make_pre_post_processors(stats: dict, device: str):
    # TODO: (pre, post) 반환
    pass
