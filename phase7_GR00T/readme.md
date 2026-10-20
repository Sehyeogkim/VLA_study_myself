## plan

GR00T (NVIDIA Isaac GR00T N1.x) 를 코드부터 뜯어보고, LIBERO 로 fine-tune -> eval.

1. 구조 읽기: LeRobot policies/groot (N1.7: Cosmos-Reason2-2B backbone + DiT action head, flow matching)
2. 데이터: lerobot/libero (data/libero, 이미 받아둠)
3. fine-tune (RunPod)
4. eval: LIBERO (phase6 eval 환경 재사용)
