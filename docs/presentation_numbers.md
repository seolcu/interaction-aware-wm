# 발표용 핵심 수치 (11/26 최종 발표)

발표 자료에 바로 옮겨 쓸 수치를 모은다. 자세한 설정, 명령, seed별 값은 `docs/experiments.md`의 해당 E 번호에 있다. 진행 중인 실험은 "(진행 중)"으로 표시하고, 결과가 나오는 대로 이 문서를 갱신한다(마지막 갱신: 11/24 23시).

## 1. 평가 방법 수정 (CF-JEPA 자체 성능)

| 수정 | 조건 | 수정 전 | 수정 후 | E |
|---|---|---|---|---|
| 첫 관측 버그 수정 | Collision TwoRoom | | CF-JEPA +11%p | E26 |
| CEM action 범위 제한 | Collision TwoRoom (r1x) | 60.8 | 73.4 (+12.6 ± 1.8, 16 seed) | E41 |
| CEM action 범위 제한 | **Distractor TwoRoom N=2 (기존 benchmark)** | 84.8 | **95.4** (16승 0패, 논문 0.95 재현) | E44 |

## 2. 제안 방법이 CF-JEPA를 이긴 결과 (평가 때 화면만 사용)

제안 방법: 화면 두 장에서 obstacle 위치·속도를 읽는 **탐지기**와 obstacle **궤적 예측기**를 obstacle별 **interaction module(h_int)**에 결합한다. 학습 때만 obstacle 상태 label을 쓴다. 모든 비교는 같은 seed로 짝지은 성공률(%) 차이 ± 표준오차다.

| 조건 | CF-JEPA | **제안** | 정답 입력 (상한) | 제안 − CF-JEPA | 이긴/진 seed | E |
|---|---|---|---|---|---|---|
| obstacle 2개 (학습한 조건, 32 seed) | 79.5 | **85.6** | 88.2 | **+6.1 ± 1.1** | 25 / 5 | E56 |
| obstacle 3개 (학습 때 안 봄, 16 seed) | 70.0 | **79.0** | 79.8 | **+9.0 ± 1.1** | 16 / 0 | E56 |
| obstacle 3개 (안 봄, 새 seed 16개) | 69.6 | 76.4 | 77.5 | +6.8 ± 1.8 | 10 / 1 | E57 |
| obstacle 4개 (안 봄, 32 seed) | 64.8 | 68.4 | 68.4 | +3.7 ± 1.5 | 20 / 11 | E57 |
| 작은 obstacle 2개, 덜 겹침 (32 seed) | 88.2 | 91.7 | 93.6 | +3.4 ± 1.0 | 25 / 7 | E58 |
| 작은 obstacle 3개, 덜 겹침 (안 봄, 16 seed) | 81.6 | 84.4 | 83.2 | +2.8 ± 1.6 (미달) | 10 / 4 | E58 |
| 평가 설정 고른 뒤 obstacle 2개 (새 seed 32개) | | | | (진행 중) | | E60 |
| obstacle 3개로 학습 (32 seed) | | | | (진행 중) | | E61 |

- **탐지기 정확도:** 속도 오차 0.2~0.3 px/step, 위치 오차 0.03~0.05px다(E56·E58). obstacle 속도는 3~6 px/step이다.
- **충돌 예측(contact z^c 오차 감소):** 제안 56~61%, 정답 입력 61~65%다. 예전 label 없는 방법(E45~E50)은 최대 20%였다.

## 3. 왜 되는가: 진단 결과

| 질문 | 결과 | E |
|---|---|---|
| CF-JEPA latent에 부족한 것 | obstacle 속도(오차 1.7 px/step). 위치는 1.8px로 담겨 있다 | E38 |
| 정답 obstacle 정보의 가치 (상한) | obstacle 1·2·3개에서 +6.5 / +10~12 / +9.6%p | E42, E47~E50, E54 |
| 속도를 얼마나 정확히 알아야 하나 | 정답 속도에 2 px/step 오차만 섞어도 이득이 0이다(+7.3 → +0.3%p) | E52 |
| label 없는 방법들의 결과 | 충돌 예측 개선 6~20%, 성공률 이득 없음 | E45~E50 |

발표 그림: `outputs/e52/fig/contact_vs_planning_gain.png`(Proton Drive `results/E52_velocity_noise_curve/`)는 충돌 예측 개선 대비 성공률 이득을 보여 준다.

## 4. 한계 (발표에서 밝힐 것)

- **학습 때 obstacle label을 쓴다.** CF-JEPA는 label 없이 학습한다. 평가 때는 둘 다 화면만 쓴다. agent 위치 cost의 probe는 두 모델이 같은 것(위치 label로 학습)을 쓴다.
- **obstacle 겹침:** 환경의 crossing spawn과 좁은 방(폭 49px) 때문에 obstacle 여러 개가 대부분의 시간 겹친다(E57). 그래서 "obstacle 3·4개 일반화"는 겹친 obstacle 묶음에 대한 결과다. 덜 겹치는 작은 obstacle에서는 학습한 개수(2개)에서만 이겼다(E58).
- 성공률은 50 episode × seed 16~32개의 평균이다. 같은 설정에서도 seed마다 ±4~8%p 흔들린다.
