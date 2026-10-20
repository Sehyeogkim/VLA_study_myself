'''
eval.py — LIBERO 시뮬레이터에서 action chunk 를 실행하고 성공률을 센다.

학습 때와 같은 관찰로 맞춘다.
    - 사진: raw 시뮬레이터는 뒤집혀 있다. 위아래·좌우를 뒤집어 lerobot/libero 영상과 같게 한다.
    - state 8: 손끝 xyz + axis-angle + 손가락 2. quat 은 axis-angle 로 바꾼다.
    - 제어: 데이터 fps 10 이라 env 한 스텝을 0.1초(control_freq=10)로 둔다.
    - chunk 40 중 앞 n_action_steps 만 실행하고 다시 예측한다.

실행은 RunPod (시뮬레이터 + GPU).
    cd phase7_GR00T
    python eval.py --ckpt result/best.pt --suite libero_spatial --task_id 0 --n_episodes 10
'''
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from configuration import Config
from modeling import GrootPolicy

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "libero"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SUITES = {
    "libero_10": (0, 10),
    "libero_goal": (10, 20),
    "libero_object": (20, 30),
    "libero_spatial": (30, 40),
}
MAX_STEPS = {
    "libero_spatial": 280,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(Path(__file__).parent / "result" / "best.pt"))
    p.add_argument("--suite", default="libero_spatial", choices=list(SUITES))
    p.add_argument("--task_id", type=int, default=None, help="스위트 안 번호 0~9. 비우면 10개 전부.")
    p.add_argument("--n_episodes", type=int, default=10, help="태스크마다 init state 를 몇 개 돌릴지")
    p.add_argument("--n_action_steps", type=int, default=None)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--n_videos", type=int, default=2)
    p.add_argument("--out_dir", default=str(Path(__file__).parent / "result" / "eval"))
    return p.parse_args()


def quat_to_axisangle(quat):
    '''(x, y, z, w) -> axis-angle 3. LeRobot LiberoProcessorStep 와 같은 식.'''
    quat = np.asarray(quat, dtype=np.float32).reshape(4)
    w = float(np.clip(quat[3], -1.0, 1.0))
    den = np.sqrt(max(0.0, 1.0 - w * w))
    if den < 1e-10:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arccos(w)
    return (quat[:3] / den * angle).astype(np.float32)


def obs_to_model(obs):
    '''LiberoEnv pixels_agent_pos -> (image [1,2,3,H,W] 0~1, state [1,8])'''
    def one(hwc):
        img = np.asarray(hwc)
        img = img[::-1, ::-1]                       # 180도. 데이터셋 카메라 방향.
        if img.dtype != np.float32:
            img = img.astype(np.float32) / 255.0
        return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)

    image = torch.stack([one(obs["pixels"]["image"]), one(obs["pixels"]["image2"])])[None]
    rs = obs["robot_state"]
    state = np.concatenate([
        np.asarray(rs["eef"]["pos"], dtype=np.float32).reshape(3),
        quat_to_axisangle(rs["eef"]["quat"]),
        np.asarray(rs["gripper"]["qpos"], dtype=np.float32).reshape(2),
    ])
    return image.to(DEVICE), torch.from_numpy(state)[None].to(DEVICE)


def load_policy(ckpt_path):
    cfg = Config()
    stats = json.loads((DATA_DIR / "meta" / "stats.json").read_text())
    policy = GrootPolicy(cfg, stats)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    policy.dit.load_state_dict(ckpt["dit"])
    policy.state_mean.copy_(ckpt["state_mean"])
    policy.state_std.copy_(ckpt["state_std"])
    policy.action_mean.copy_(ckpt["action_mean"])
    policy.action_std.copy_(ckpt["action_std"])
    print(f"loaded {ckpt_path} (step {ckpt.get('step')}, val {ckpt.get('val')})", flush=True)
    return policy.to(DEVICE).eval(), cfg


def rollout(env, policy, n_action_steps, max_steps, save_video):
    obs, info = env.reset()
    queue, frames = [], []
    success = False
    text = env.task_description
    for step in range(max_steps):
        if not queue:
            image, state = obs_to_model(obs)
            chunk = policy.sample_actions(image, [text], state)[0]          # [40, 7]
            queue = [a.astype(np.float32) for a in chunk[:n_action_steps].detach().cpu().numpy()]
        action = np.clip(queue.pop(0), -1.0, 1.0)
        obs, reward, terminated, truncated, info = env.step(action)
        if save_video:
            frames.append(env.render())
        if info.get("is_success", False):
            success = True
        if terminated or truncated:
            break
    return success, step + 1, frames


def main():
    args = parse_args()
    if DEVICE != "cuda":
        raise SystemExit("cuda 없음. LIBERO eval 은 RunPod 에서 돌린다.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    policy, cfg = load_policy(args.ckpt)
    n_action_steps = args.n_action_steps or cfg.n_action_steps
    max_steps = args.max_steps or MAX_STEPS[args.suite]

    from lerobot.envs.libero import LiberoEnv, _get_suite
    suite = _get_suite(args.suite)
    task_ids = [args.task_id] if args.task_id is not None else list(range(len(suite.tasks)))

    results = []
    t0 = time.time()
    videos_left = args.n_videos
    for task_id in task_ids:
        env = LiberoEnv(
            task_suite=suite,
            task_id=task_id,
            task_suite_name=args.suite,
            obs_type="pixels_agent_pos",
            control_freq=cfg.fps,               # 10 Hz. 데이터 한 칸 = 0.1초.
            init_states=True,
            episode_index=0,
        )
        print(f"task {task_id}: {env.task_description}", flush=True)
        for ep in range(args.n_episodes):
            save = videos_left > 0
            success, steps, frames = rollout(env, policy, n_action_steps, max_steps, save)
            results.append({
                "suite": args.suite, "task_id": task_id, "episode": ep,
                "success": success, "steps": steps, "task": env.task_description,
            })
            print(f"  ep {ep:02d} | success {success!s:5} | steps {steps}", flush=True)
            if save and frames:
                import imageio.v2 as imageio
                name = f"{args.suite}_t{task_id:02d}_ep{ep:02d}_{'success' if success else 'fail'}.mp4"
                imageio.mimsave(out_dir / name, frames, fps=cfg.fps)
                videos_left -= 1
        env.close()

    sr = float(np.mean([r["success"] for r in results])) if results else 0.0
    summary = {
        "ckpt": args.ckpt, "suite": args.suite, "n": len(results),
        "n_action_steps": n_action_steps, "success_rate": sr,
        "seconds": time.time() - t0,
    }
    json.dump({"summary": summary, "episodes": results}, open(out_dir / "eval.json", "w"), indent=1)
    print("================================================")
    print(f"success rate {sr * 100:.1f}% | {len(results)} episodes | {args.suite}")
    print("saved:", out_dir)


if __name__ == "__main__":
    main()
