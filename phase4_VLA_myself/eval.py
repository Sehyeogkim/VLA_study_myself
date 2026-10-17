'''
eval.py — PushT 시뮬레이터에서 rollout -> 성공률 / coverage  (phase2 eval.py 기반)

phase2 와 다른 점:
    - 모델: SmolVLA_overall (best.pt 의 state_dict, 정규화 buffer 포함) -> sample_actions
    - 이미지 2 frame: env 는 현재 frame 만 줌 -> 직전 frame 을 기억해서 [이전, 현재] 로 묶음
      (데이터셋 delta -0.1 초 = env 1 step 전. episode 첫 step 은 현재 frame 두 번 = 학습 때 pad 처리와 같음)

흐름 (episode 마다):
    obs = env.reset(seed)
    반복 (최대 max_steps):
        action queue 가 비면: [이전, 현재] frame + state -> sample_actions -> chunk[:n_action_steps] 를 queue 에
        queue 에서 1 개 꺼내 env.step(action), 이전 frame 갱신
점수:
    success  : env 가 주는 is_success (T 블록이 목표와 충분히 겹침)
    coverage : episode 동안 reward(= 겹친 비율) 의 최대값

실행 (RunPod): cd phase4_VLA_myself && python eval.py --ckpt result/best.pt --n_episodes 50
'''
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from configuration import Config
from modeling import SmolVLA_overall

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TASK = "Push the T-shaped block onto the T-shaped target."     # 학습 데이터의 task 문장 그대로
ROOT = Path(__file__).resolve().parents[1]                     # VLA_study_myself/


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=str(Path(__file__).parent / "result" / "best.pt"))
    p.add_argument("--n_episodes", type=int, default=50)
    p.add_argument("--n_action_steps", type=int, default=8)     # chunk 16 개 중 앞 8 개만 실행하고 다시 예측
    p.add_argument("--max_steps", type=int, default=300)        # LeRobot PushT episode_length
    p.add_argument("--seed", type=int, default=1000)            # episode i 는 seed + i (phase2 와 같은 문제)
    p.add_argument("--n_videos", type=int, default=4)           # 앞 N 개 episode 영상 저장
    p.add_argument("--out_dir", type=str, default=str(Path(__file__).parent / "result" / "eval"))
    return p.parse_args()


def load_policy(ckpt_path):
    cfg = Config()
    # stats 는 생성자 모양 맞추기용 -> 바로 아래 load_state_dict 가 학습 때 저장한 mean / std buffer 로 덮어씀
    stats = json.load(open(ROOT / "data" / "pusht" / "meta" / "stats.json"))
    model = SmolVLA_overall(cfg, stats)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    print(f"loaded {ckpt_path} (step {ckpt['step']}, val {ckpt['val']:.4f})", flush=True)
    return model.to(DEVICE).eval(), cfg


def obs_to_img(obs):
    '''env 관측 이미지 -> [3, 96, 96] 0~1 (데이터셋 이미지와 같은 형식)'''
    img = torch.from_numpy(obs["pixels"]).permute(2, 0, 1)[None].float() / 255.0    # [1, 3, H, W]
    if img.shape[-1] != 96:
        img = F.interpolate(img, size=(96, 96), mode="bilinear", align_corners=False)
    return img[0].to(DEVICE)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step1 : 모델
    model, cfg = load_policy(args.ckpt)

    # Step2 : env (gym-pusht)
    import gymnasium as gym
    import gym_pusht  # noqa: F401  (gym 에 PushT-v0 등록)
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array",
                   max_episode_steps=args.max_steps)
    obs, _ = env.reset(seed=args.seed)
    print("obs pixels", obs["pixels"].shape, obs["pixels"].dtype, "| agent_pos", obs["agent_pos"],
          "| action space", env.action_space, flush=True)

    # Step3 : episode 반복
    results = []
    t0 = time.time()
    for ep in range(args.n_episodes):
        obs, info = env.reset(seed=args.seed + ep)
        queue, frames, max_reward, success = [], [], 0.0, False
        save_video = ep < args.n_videos
        prev_img = obs_to_img(obs)                       # 첫 step: 이전 frame 없음 -> 현재 frame 으로 채움

        for step in range(args.max_steps):
            cur_img = obs_to_img(obs)
            if not queue:
                image = torch.stack([prev_img, cur_img])[None]                                   # [1, 2, 3, 96, 96]
                state = torch.from_numpy(np.asarray(obs["agent_pos"], dtype=np.float32))[None].to(DEVICE)   # [1, 2]
                chunk = model.sample_actions(image, [TASK], state)[0]                           # [16, 2] 픽셀 좌표
                queue = list(chunk[: args.n_action_steps].cpu().numpy())
            action = queue.pop(0)

            prev_img = cur_img                           # 매 env step 마다 갱신 (0.1 초 전 = 1 step 전)
            obs, reward, terminated, truncated, info = env.step(action)
            max_reward = max(max_reward, float(reward))
            if save_video:
                frames.append(env.render())
            if info.get("is_success", False):
                success = True
            if terminated or truncated:
                break

        results.append({"episode": ep, "seed": args.seed + ep, "success": success,
                        "coverage": max_reward, "steps": step + 1})
        print(f"ep {ep:3d} | success {success!s:5} | coverage {max_reward:.3f} | steps {step + 1}", flush=True)

        if save_video and frames:
            import imageio
            imageio.mimsave(out_dir / f"ep{ep:03d}_{'success' if success else 'fail'}.mp4", frames, fps=10)

    # Step4 : 점수
    sr = float(np.mean([r["success"] for r in results]))
    cov = float(np.mean([r["coverage"] for r in results]))
    summary = {"ckpt": args.ckpt, "n_episodes": args.n_episodes, "n_action_steps": args.n_action_steps,
               "success_rate": sr, "mean_coverage": cov, "seconds": time.time() - t0}
    json.dump({"summary": summary, "episodes": results}, open(out_dir / "eval.json", "w"), indent=1)
    print("================================================")
    print(f"success rate {sr * 100:.1f}% | mean coverage {cov:.3f} | {args.n_episodes} episodes")
    print("saved:", out_dir)


if __name__ == "__main__":
    main()
