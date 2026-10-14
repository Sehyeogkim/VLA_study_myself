"""
eval.py — 평가 루프 (lerobot: scripts/lerobot_eval.py)

흐름:
    env.reset() -> obs -> pre(obs) -> policy.select_action -> post(action)
    -> env.step(action) -> 반복 -> success 여부 기록 -> 성공률
"""
import argparse

import torch

from modeling import MyPolicy
from processor import make_pre_post_processors


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--n_episodes", type=int, default=50)
    p.add_argument("--max_steps", type=int, default=300)
    return p.parse_args()


def main():
    args = parse_args()
    # TODO Step1: gym-pusht env 생성
    # TODO Step2: checkpoint 로드 (policy, pre, post)
    # TODO Step3: episode 루프 (매 episode 마다 action queue reset)
    # TODO Step4: 성공률 / 평균 reward 출력 + rollout 영상 저장
    pass


if __name__ == "__main__":
    main()
