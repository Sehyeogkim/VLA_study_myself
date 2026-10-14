# Phase 2 — Policy Myself (Flow Matching on PushT)

SmolVLA 구조를 참고해서 작은 flow-matching 정책을 **원본 코드 안 보고** 직접 구현.

## LeRobot policy 공통 구조
ACT, Diffusion, Pi0, SmolVLA, GR00T 등 21개 policy 전부 같은 틀 (프랜차이즈: 메뉴는 달라도 매장 구조는 같음)

```
configuration_xxx.py   모델 하이퍼파라미터 + optimizer preset
modeling_xxx.py        아키텍처 + loss + action 생성
processor_xxx.py       normalize / unnormalize (데이터 손질만)
```
- modeling 은 2층
  - 바깥 **Policy** (`PreTrainedPolicy`): `forward`(loss), `select_action`(queue), `predict_action_chunk`
  - 안쪽 **Model** (`nn.Module`): 실제 신경망 (`VLAFlowMatching`, `PI0Pytorch`, `ACT` ...)
- 학습/평가 **루프는 별도 스크립트** (`lerobot_train.py`, `lerobot_eval.py`) → 같은 인터페이스만 호출하므로 `--policy.type` 만 바꾸면 어떤 모델이든 동작
- 하이퍼파라미터 위치
  - 모델 쪽 (chunk_size, num_steps, lr) → `configuration`
  - 실험 쪽 (batch_size, steps, num_workers) → train config / argparse

## Files
| file | 역할 | smolvla 대응 |
| --- | --- | --- |
| `configuration.py` | 모델 하이퍼파라미터 | `configuration_smolvla.py` |
| `modeling.py` | `MyPolicy`(바깥) + `MyFlowMatchingModel`(안쪽) | `modeling_smolvla.py` |
| `processor.py` | normalize / unnormalize | `processor_smolvla.py` |
| `train.py` | 학습 루프 | `lerobot_train.py` |
| `eval.py` | PushT rollout → 성공률 | `lerobot_eval.py` |

## Plan
- [ ] PushT 데이터 구조 확인 (`lerobot/pusht`)
- [ ] configuration
- [ ] processor
- [ ] modeling (embed → forward loss → sample_actions)
- [ ] train
- [ ] eval → **success rate**
- [ ] Diffusion Policy / ACT 와 비교

## Results
(TBD)
