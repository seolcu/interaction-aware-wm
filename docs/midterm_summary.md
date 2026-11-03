# 중간 발표(11/5) 결과 요약

발표 자료 작성용 요약이다. 프로포절(10/12) 이후 바뀐 것만 담는다. 세부 설정과 수치의 근거는 `docs/experiments.md`의 E 번호를 따라가면 된다. 수치는 모두 planning success rate(%)이고, 따로 적지 않으면 50 episodes, seed 1개다.

## 한 줄 요약

CF-JEPA를 직접 재구현해 기존 benchmark 두 개(TwoRoom, PushT)에서 논문의 핵심 결과를 재현했다. 우리가 만든 Collision env에서는 obstacle이 많아질수록 CF-JEPA 성능이 떨어져(91% → 65%) 독립 가정의 한계를 수치로 보였다. 제안한 interaction gate는 충돌 순간에 정확히 켜지도록 학습되지만, 아직 성공률을 올리지는 못했다. 그 원인을 encoder의 obstacle 정보 부족으로 좁혀 가는 중이다.

## 1. 기존 benchmark 재현 (교수님 요청: 기존 task 수치)

모든 모델은 distractor 사각형 2개(N=2)를 덧그린 화면으로 학습하고, N=2 환경에서 평가했다(논문 Table III와 같은 조건).

| Task | LeWM (우리) | LeWM (논문) | CF-JEPA (우리) | CF-JEPA (논문) | 근거 |
|---|---|---|---|---|---|
| TwoRoom | 30 | 33 | 88 (cost 튜닝 시 99, seed 4개 평균) | 95 | E3, E5, E6 |
| PushT | 2 | 1 | 70 (cost 튜닝 시 78) | 87 | E7 |

- 두 task 모두 "distractor로 학습하면 LeWM은 무너지고 CF-JEPA는 버틴다"는 논문의 결론을 재현했다.
- LeWM은 깨끗한 화면(N=0)에서 평가해도 TwoRoom 34%, PushT 2%다. 학습 중에 latent가 사각형 움직임까지 담아 planning이 망가진 것으로 해석한다.
- CF-JEPA 코드는 공개되지 않아 논문만 보고 직접 구현했다(E3). PushT가 논문보다 낮은 것은 seed 1개(표준오차 약 ±6%p)와 구현 세부(distractor 배치 등)의 차이로 본다.
- "cost 튜닝"은 planning cost를 논문 Eq. 7(`sum_l2`)에서 LeWM 방식(`last_mse`)으로 바꾼 것이다. 재학습은 없다.

## 2. Collision env (우리가 만든 환경)

- TwoRoom에 움직이는 obstacle을 더해, obstacle이 agent를 밀어내게 했다(E1). CF-JEPA가 가정하는 "조종할 수 없는 것은 조종할 수 있는 것에 영향을 주지 않는다"가 깨지는 환경이다.
- 학습 data 1만 episode를 모았다(E4). Episode의 73%에서 충돌이 일어나고, 충돌 step은 전체의 7%다.

## 3. Interaction gate: 결과와 실패 원인 추적

제안 모델은 CF-JEPA의 예측에 `ẑ^c = g + α_t · h_int(z^c, z^u, a)`를 더한다. Gate `α_t`가 충돌 때만 켜져서 h_int가 obstacle의 영향을 보정한다는 아이디어다.

| 단계 | 한 일 | 결과 | 근거 |
|---|---|---|---|
| 1 | Gate 첫 학습(sparsity 방식) | gate가 항상 꺼지거나 항상 켜짐 | E8 |
| 2 | Gate 학습 신호 바꾸기 | 충돌 라벨로 학습하면 충돌 때 정확히 켜짐(AUROC 0.90). 그런데 켜도 예측이 0.8%만 좋아짐 | E9 |
| 3 | 원인 찾기(잔차 probe) | 정답 obstacle 위치가 있으면 충돌 오차의 40%를 설명하지만, 모델의 latent로는 4.6%만 설명 → **encoder가 obstacle을 흐릿하게 담는 것이 병목** | E9 |
| 4 | Encoder 고치기 시도 | 충돌 라벨 2%를 보조 신호로 주면 조금 나아짐(oracle 대비 30% → 44%). **distractor를 없애면 oracle만큼 담음(99%)** → 사각형과 obstacle이 latent 용량을 두고 경쟁 | E10, E11 |
| 5 | 고친 encoder로 gate 재학습 | gate는 여전히 충돌 때 잘 켜지지만(AUROC 0.90) 예측 개선은 0.5% | E12 |
| 6 | 성공률 평가 | Collision env: gate 모델 91% = CF-JEPA 91%. Distractor TwoRoom: gate 모델 90~94%로 robustness 유지 | E13 |

- 관련 최신 연구(arXiv 2606.30068)도 "reward 없이 예측만 학습하면 control에 중요한 외부 요소의 정보를 버리고, 라벨 약 2%로 되살릴 수 있다"고 보고했다. 우리 관찰(3, 4단계)과 같은 방향이다.

## 4. Obstacle이 많아지면? (교수님 질문)

학습은 obstacle 1개 data로 하고, 평가만 어렵게 했다(E14, 조건당 100 episodes).

| 평가 조건 | CF-JEPA | Gate 모델 |
|---|---|---|
| Obstacle 1개 (기본) | 91 | 91 |
| Obstacle 3개 | 73 | 71 |
| Obstacle 3개 + 시간 제한 짧게 | **65** | 66 |
| Obstacle 2개 + 빠르게 | 73 | 82 (gate를 꺼도 82) |

- **Obstacle이 많아질수록 CF-JEPA가 떨어진다.** CF-JEPA 저자가 §V에 적은 한계가 실제로 성능 저하로 나타난다는 것을 처음으로 수치로 보였다.
- 현재 gate는 obstacle이 많아져도 차이를 만들지 못한다. Gate가 켜져도 h_int가 충돌 효과를 배우지 못했기 때문이다(3장 5단계).
- E15: obstacle 3개 data로, distractor 없이(4단계에서 encoder가 obstacle을 가장 잘 담은 조건) CF-JEPA와 gate를 10 epoch 다시 학습했다(평가 150 episodes).

| 평가 조건 | CF-JEPA | Gate 모델 (gate 켬) | 같은 모델 gate 끔 |
|---|---|---|---|
| Obstacle 3개 | 78.0 | 80.7 | 73.3 |
| Obstacle 3개 + 시간 제한 | 60.7 | 59.3 | 60.0 |

- Gate는 충돌 순간을 더 정확히 판별했지만(AUROC 0.94), 켰을 때 예측 오차 감소는 0.7%로 거의 그대로였다. CF-JEPA 대비 2.7%p 차이는 오차 범위(약 ±3.3%p) 안이다. distractor를 없애는 것만으로는 h_int가 충돌 효과를 배우지 못했다.

## 5. 배운 것과 최종 발표까지 할 일

- **배운 것:** gate라는 "스위치"는 쉽게 학습된다. 어려운 것은 스위치가 켜졌을 때 쓸 정보, 즉 obstacle의 정확한 위치와 속도를 encoder가 latent에 담게 하는 것이다. 예측 loss만으로는 이 정보가 학습 중에 점점 사라진다(3 epoch 30% → 10 epoch 11%).
- **최종 발표까지:**
  - h_int의 예측 이득이 작은 원인 확인(정답 obstacle 상태를 h_int에 직접 넣는 상한 실험, h_int 용량)
  - Obstacle 개수별 성능 곡선(1~5개)
  - Seed 3개 반복과 평가 episode 늘리기

## 발표 자료에 쓸 만한 그림·영상 (Proton Drive `results/`)

- `E1_collision_tworoom/`: Collision env 화면
- `E3_cfjepa_tworoom/eval/`, `E7_pusht/eval/`: 평가 영상(agent / goal 비교)
- `pusht_distractor_sheet.png`: distractor가 덧그려진 PushT 화면
