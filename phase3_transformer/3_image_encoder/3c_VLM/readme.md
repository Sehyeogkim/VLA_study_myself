## Procedure
    0. 데이터 한 샘플 열어보기       ← 짧게. 입력/정답이 뭔지만 확정
    1. modeling + 추론 (generate)    ← 랜덤/pretrained 가중치로 shape·mask·cache 검증
    2. train.py                      ← Dataset/loader, loss(label 마스킹)
    3. 작은 배치 overfit 테스트       ← loss 가 0 근처로 가는지 (phase2 에서 했던 것)
    4. 전체 학습 (RunPod)
    5. eval.py                         ← 정답 캡션이랑 나란히 출력 + 속도 비교