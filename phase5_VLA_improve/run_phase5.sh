#!/bin/bash
# phase5 실험 1 (random crop + EMA) : RunPod pod 안에서 실행
#   overfit 테스트 -> 본 학습 20k -> 저장된 checkpoint 들 eval (50 episode 씩) 까지 한 번에
# 로그는 /workspace/phase5.log + RunPod 웹 Logs 탭 (/proc/1/fd/1) 에 같이 찍힘
cd /workspace/phase5_VLA_improve
FILTER="torchcodec|ffmpeg|libav|HF_TOKEN|UNEXPECTED|_dlopen|load_library|ctypes|^ +File|^ +self|^ +torch|^ +raise|Traceback|OSError|exception was the direct"

echo "=== 1. overfit test (300 step) ==="
python -u train.py --overfit --steps 300 2>&1 | grep --line-buffered -v -i -E "$FILTER"
mkdir -p result && mv result/history.json result/overfit_history.json 2>/dev/null

echo "=== 2. train 20k ==="
python -u train.py 2>&1 | grep --line-buffered -v -i -E "$FILTER"

echo "=== 3. eval (50 episode, n_action_steps 8) ==="
for ck in best best_ema step5000_ema step10000_ema step15000_ema step20000_ema step20000; do
  if [ -f result/$ck.pt ]; then
    echo "--- $ck ---"
    python -u eval.py --ckpt result/$ck.pt --n_episodes 50 --n_videos 2 --out_dir result/eval_$ck 2>&1 \
      | grep --line-buffered -E "loaded|success rate|Error"
  fi
done
echo "=== ALL DONE ==="
