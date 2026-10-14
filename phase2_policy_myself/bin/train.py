"""
train.py — 학습 루프 (lerobot: scripts/lerobot_train.py, phase1: 05_fine_tuning.py)

흐름:
    dataset -> DataLoader -> pre(batch) -> policy.forward(batch) -> loss.backward
    -> clip grad -> optimizer.step -> scheduler.step -> log / val / checkpoint
"""
import argparse

import torch

from configuration import MyPolicyConfig
from modeling import MyPolicy
from processor import make_pre_post_processors


def parse_args():
    p = argparse.ArgumentParser()
    # 실험 쪽 하이퍼파라미터
    p.add_argument("--steps", type=int, default=10_000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--save_every", type=int, default=2_000)
    return p.parse_args()


def main():
    args = parse_args()
    # TODO Step1: dataset (lerobot/pusht) + train/val split by episode
    # TODO Step2: DataLoader + next_batch (iter 소진되면 재생성)
    # TODO Step3: config, policy, pre/post (dataset stats 넣기!)
    # TODO Step4: optimizer (AdamW) + scheduler (warmup + cosine)
    # TODO Step5: loop -> loss, grad clip, step, log
    # TODO Step6: checkpoint 저장 + loss 그래프
    pass


if __name__ == "__main__":
    main()
