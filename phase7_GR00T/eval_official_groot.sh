#!/bin/bash
# eval_official_groot.sh — NVIDIA 공식 GR00T N1.7 (LIBERO-Spatial 로 파인튜닝된 것) 을 LIBERO 에서 eval.
# 학습 X. phase6 의 SmolVLA eval (66%, Spatial 10 task x 10 episode) 과 같은 조건으로 비교한다.
#
# 모델: nvidia/gr00t17-lerobot-libero_spatial-640
#   - LeRobot 0.6.1 로 학습 (20k step, batch 320), embodiment libero_sim
#   - 입력 키: observation.images.image (3인칭) / observation.images.wrist_image (손목) / observation.state [8]
#   - LIBERO 환경은 image / image2 를 주니까 image2 -> wrist_image 로만 이름을 바꿔준다
#   - config.json 의 base_model_path 가 NVIDIA 내부 캐시 경로 (/cache/...) 라서 HF repo 이름으로 덮어씀
#   - gripper 부호 변환 (libero), 4 step Euler 는 체크포인트 설정이 알아서 함
#
# 실행 (RunPod pod 안, A40 이면 충분):
#   bash eval_official_groot.sh setup     # 처음 한 번: apt + pip (약 5 분)
#   bash eval_official_groot.sh try       # task 0 만 3 episode (환경 / 키 확인, 약 2 분)
#   bash eval_official_groot.sh spatial   # 본 eval: 10 task x 10 episode (약 30~40 분)
#
# 결과: /workspace/eval_groot_<mode>/eval_info.json  (진행바의 running_success_rate 는 버그라 무시, 이 파일이 정답)
set -e
MODE=${1:-try}
POLICY=nvidia/gr00t17-lerobot-libero_spatial-640
RENAME='{"observation.images.image2": "observation.images.wrist_image"}'

if [ "$MODE" = "setup" ]; then
  apt-get update -qq && apt-get install -y -qq tmux libegl1 libgl1 libglvnd0 libgles2 libosmesa6 cmake ffmpeg >/dev/null
  pip install --break-system-packages "lerobot[libero,groot]==0.6.1"
  echo SETUP_DONE
  exit 0
fi

export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl      # 모니터 없는 서버에서 GPU 로 렌더링
# Cosmos-Reason2-2B 는 gated repo -> HF 토큰 필요. 로컬 .env 를 /workspace/.env 로 복사해 둔 것을 읽는다 (값은 출력 X)
if [ -f /workspace/.env ]; then
  set -a; . /workspace/.env; set +a
  export HF_TOKEN="${HF_TOKEN:-$HUGGINGFACE_TOKEN}"
fi
cd /workspace

if [ "$MODE" = "try" ]; then
  EXTRA=(--eval.n_episodes=3 --env.task_ids="[0]")
elif [ "$MODE" = "spatial" ]; then
  EXTRA=(--eval.n_episodes=10)                  # SmolVLA 때와 같은 100 episode
else
  echo "mode: setup | try | spatial"; exit 1
fi

OUT=/workspace/eval_groot_$MODE
rm -rf "$OUT"
lerobot-eval \
  --policy.path=$POLICY \
  --policy.base_model_path=nvidia/GR00T-N1.7-3B \
  --env.type=libero --env.task=libero_spatial \
  --eval.batch_size=1 \
  --rename_map="$RENAME" \
  --output_dir=$OUT \
  "${EXTRA[@]}" 2>&1 | grep --line-buffered -v -E "Fetching|Loading weights|Running rollout"

python3 - <<EOF
import json
d = json.load(open("$OUT/eval_info.json"))
o = d.get("overall", d)
print("overall pc_success", o["pc_success"], "| n", o["n_episodes"])
for t in d.get("per_task", []):
    s = t["metrics"]["successes"]
    print("task", t["task_id"], sum(s), "/", len(s))
EOF
echo DONE
