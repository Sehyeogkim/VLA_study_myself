# Phase 2 — Policy Myself (작은 VLA 를 직접 만들어 PushT 에 학습)

SmolVLA 를 해부한 뒤(phase1), 같은 아이디어로 **VLM + flow-matching action expert** 정책을 원본 코드 안 보고 직접 구현.
목표: **학습 → 추론 → 시뮬레이터 점수** 한 바퀴 돌리기.

## 1. 구조 (B 방식: VLM freeze + cross-attention expert)

```
image [96×96] → 512 resize → SigLIP(vision) → connector → img tokens [64 × 960]  ┐
task  문장     → tokenizer → embed_tokens            → lang tokens [30 × 960]  ├─ prefix [95 × 960]
state [x, y]   → normalize → state_proj (Linear, 학습)  → state token [1 × 960]  ┘
prefix → text_model (Llama 32 layer, freeze, 4D attention mask) → last_hidden_state [95 × 960]
                                                                        │ memory_proj (960 → 512)
x_t [16 × 2], t → action_in_proj + sin/cos time emb → concat + MLP → suffix [16 × 512]
suffix ─ TransformerDecoder × 4 (self-attn + cross-attn → memory) → action_out_proj → v_t [16 × 2]
```

| 부분 | 내용 | 학습 |
|---|---|---|
| VLM | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` (vision + connector + text_model 32 layer) | ❌ freeze |
| `state_proj` | Linear(2 → 960) | ✅ |
| Action expert | memory_proj, action_in_proj, action_time_mlp, TransformerDecoder ×4 (D=512, 8 head), action_out_proj | ✅ |
| 학습 파라미터 | **18.1M** | |

- A 방식(SmolVLA/Pi0: layer 별 KV 공유)이 아니라 **VLM 마지막 hidden 만 cross-attention 으로 읽는 B 방식** (GR00T 계열).
- Flow matching: `x_t = t·noise + (1−t)·a`, 정답 `u_t = noise − a`, `t ~ Beta(1.5, 1)·0.999 + 0.001`, loss = MSE (pad 된 action 제외).
- 추론: noise(t=1) → Euler 10 step (dt = −0.1) → action chunk 16 개 → unnormalize.

## 2. 파일

| 파일 | 역할 |
|---|---|
| `configuration.py` | `MyPolicyConfig` — 모델 하이퍼파라미터 (chunk 16, expert D 512, num_steps 10, lr preset …) |
| `modeling.py` | `Prefix`, `ActionExpert`, `compute_loss`, `sample_actions`, `load_policy` |
| `utils.py` | `make_att_2d_masks`, `save_mask_png`, `sinusoidal_time_emb`, `BatchStream` |
| `train.py` | 학습 루프 (episode 단위 train/val 분리, warmup+cosine, grad clip, 고정 val, 체크포인트) |
| `eval.py` | gym-pusht rollout → 성공률 / coverage / 영상 |
| `check_ckpt.py` | 체크포인트 로드 검증 (val loss 재계산 + sample_actions) |
| `bin/` | 처음 작성한 원본 (`Full_script_train.py`, `Full_script_infer.py`), 초기 뼈대, `lr_schedule_viz.py` |

체크포인트(`*.pt`)는 git 에서 제외 — 로컬 `results/run1/` 에만 있음.

## 3. 데이터

`lerobot/pusht` — 206 episode, 25,650 frame, 10fps, image 96×96, state/action 2D (픽셀 좌표 0~512).
- train: episode 0~189 (23,474 frame) / val: episode 190~205 (2,176 frame)
- `delta_timestamps` 로 action chunk 16 개 (1.6 초) + `action_is_pad`
- 정규화: dataset stats 의 mean/std → 모델 buffer 로 저장 (체크포인트에 같이 들어감)

## 4. 진행 과정

1. **구조 확인** (`bin/Full_script_infer.py`): 가짜 입력으로 prefix → VLM → expert → Euler 10 step shape 확인. mask PNG 로 attention 규칙 확인.
2. **Overfit 테스트** (batch 4 고정, 200 step, Mac MPS): loss 3.66 → ~0.3 → 파이프라인/gradient 정상.
   - warmup 없이 Adam 초반 loss spike (최대 48) → **warmup 추가로 해결** (grad clip 만으로는 안 잡힘).
   - 결과: `results/train_overfit.png`, lr 모양: `results/lr_schedule.png`
3. **RunPod 학습 run1** (A40):
   - batch 64, lr 1e-4, warmup 300 + cosine (6,000 step 기준), val 500 step 마다 (고정 batch 8 개 + 고정 noise/t)
   - 0.55 it/s, GPU 메모리 14GB, step 5,300 에서 중단 (~3h20m, ~$2)
4. **구조 정리**: Full_script → `modeling.py` / `train.py` / `eval.py` 분리 (속성 이름 유지 → 기존 체크포인트 그대로 로드).
5. **체크포인트 검증** (RunPod): `best.pt` 로드 → val loss 0.4533 (로그 0.4464 와 일치 수준) ✅
6. **PushT 평가** (RunPod, gym-pusht, 10 episode, n_action_steps 8, max 300 step)

## 5. 결과

### 학습 (run1) — `results/run1/loss_lr.png`

| step | 0 | 500 | 1000 | 2000 | 3000 | **4500 (best)** | 5000 |
|---|---|---|---|---|---|---|---|
| val loss | 3.61 | 0.61 | 0.52 | 0.48 | 0.456 | **0.446** | 0.448 |

- 3,000 step 이후 val 은 ~0.45 에서 평평, train 은 0.3~0.4 로 계속 내려감 → 이 구조/데이터로는 val ~0.45 가 한계에 가까움.

### 평가 — `results/eval_run1/` (eval.json, mp4, strip.png)

| 정책 | 성공률 | 평균 coverage |
|---|---|---|
| **내 모델 (best.pt, step 4500)** | **0 / 10 (0%)** | **0.177** |
| 랜덤 정책 (같은 seed) | 0 / 10 | 0.086 |

- 영상: 에이전트가 **T 블록에 다가가서 밀기는 함** (랜덤보다 coverage 2 배). 하지만 **목표 방향/각도로 밀지 못하고** 화면 구석으로 밀어내는 경우가 많음.
- `sample_actions` 예측 chunk 가 step 마다 ~30px 씩 튐 (정답 대비 평균 오차 42px).

## 6. 원인 추정 (아직 검증 안 함)

1. **비전 인코더 freeze** (가장 유력): 96×96 을 512 로 키워 범용 VLM 에 넣음 → T 블록의 정확한 위치/각도가 hidden 에 잘 안 담길 수 있음. Diffusion Policy 는 ResNet18 을 **같이 학습**.
2. **관측 1 frame**: 블록이 움직이는 방향을 모름 (Diffusion Policy 는 2 frame).
3. **loss 가 아직 높음**: val 0.45 에서 정체 → action chunk 가 매끄럽지 않음.
4. **학습량 부족**: 5,300 step (Diffusion Policy 는 훨씬 길게 학습).
5. 매 step VLM 500M forward + backward (state_proj 때문) → 0.55 it/s 로 느림.

## 7. 다음 시도 (후보)

- [ ] **비전 쪽 학습 허용**: SigLIP 마지막 몇 layer / connector unfreeze, 또는 작은 ResNet18 인코더로 교체 (Diffusion Policy 와 같은 조건)
- [ ] **속도 개선**: state 를 expert 쪽으로 옮기고 VLM 출력 캐싱 (VLM 이 완전 고정이 됨) 또는 bf16 autocast
- [ ] 관측 2 frame (`delta_timestamps` 에 image/state -0.1s 추가)
- [ ] 이미지 augmentation (random crop)
- [ ] 더 긴 학습 + 50 episode 평가
- [ ] 비교 기준: `lerobot/diffusion_pusht` 를 같은 `eval.py` 조건으로 평가
- [ ] A 방식 (SmolVLA 식 layer 별 KV 공유) 과 비교

## 8. 실행 (RunPod 에서)

```bash
# 학습
python phase2_policy_myself/train.py --batch_size 64 --num_workers 8 --steps 6000 --warmup 300 \
    --overfit 0 --val_every 500 --save_every 2000 --val_batches 8 --out_dir results/run2
# 체크포인트 검증
python phase2_policy_myself/check_ckpt.py
# 평가
python phase2_policy_myself/eval.py --ckpt phase2_policy_myself/results/run1/best.pt --n_episodes 50
```
환경: lerobot @ b9cb121 (`[smolvla,dataset,pusht]`), imageio-ffmpeg, A40 이미지 `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`.
