# Interaction-Aware CF-JEPA: Learning When Uncontrollable Objects Matter

**상호작용 인지 CF-JEPA: 조종할 수 없는 물체가 중요해지는 순간 학습하기**

아주대학교 2026-2학기 인공지능 팀 프로젝트 (17팀, 발표 2그룹)

평소엔 무시해도 되는 움직이는 물체가 로봇과 부딪히는 순간, 월드 모델이 그 물체를 중요한 상태 정보로 바꿔 예측하게 만드는 연구입니다.

## 연구 질문

- **RQ1.** CF-JEPA에 상호작용 게이트를 더하면, 충돌 상황에서 예측과 계획 성공률이 회복되는가?
- **RQ2.** 게이트는 실제로 부딪히는 순간에만 열리는가?

## 배경

- **LeWM** ([arXiv 2603.19312](https://arxiv.org/abs/2603.19312))
  - 픽셀에서 바로 학습하는 15M 규모의 JEPA 월드 모델이다.
  - 붕괴 방지는 SIGReg, 계획은 CEM으로 한다.
- **CF-JEPA** ([arXiv 2610.00727](https://arxiv.org/abs/2610.00727))
  - LeWM의 잠재를 조종 가능한 `z^c`(128차원)와 조종 불가한 `z^u`(64차원)로 나눠, 화면 방해물에 강해졌다.
  - 두 부분을 나누는 장치는 inverse dynamics 손실과 gradient reversal 기반 adversarial 손실이다.
  - 코드는 공개되지 않았다.
- **한계 (CF-JEPA §V, 원문):**
  > "we assume that the uncontrollable part of the scene never affects the controllable part"
  - 예측기 `g(z^c, a)`는 `z^u`를 입력으로 받지 않는다.
  - 실험의 방해물도 로봇과 물리적으로 부딪히지 않는다.

## 제안 방법

```
ẑ^c_{t+1} = g(z^c_t, a_t) + α_t · h_int(z^c_t, z^u_t, a_t)
α_t       = σ(MLP([z^c_t, z^u_t, a_t]))
L         = L_CF-JEPA + ρ · mean(α_t)        # 희소성: 평소엔 게이트를 닫게
```

- **α ≈ 0일 때:** 모델이 CF-JEPA와 같아지므로 방해물 강건성이 유지된다.
- **충돌할 때:** 예측 손실이 게이트를 열어 `z^u` 정보를 반영한다.

## 실험 계획

| 항목 | 내용 |
|---|---|
| 환경 ① | CF-JEPA와 같은 색 사각형 방해물. TwoRoom, PushT. 강건성이 유지되는지 확인 |
| 환경 ② | **새로 제작.** TwoRoom에 움직이는 장애물을 넣고, 닿으면 로봇을 밀거나 막게 함 |
| 비교군 | LeWM / CF-JEPA (재구현) / **CF-JEPA + 게이트** |
| Ablation | α=1 고정, 희소성 제거, `z^u` stop-gradient |
| 지표 | 계획 성공률, 충돌 전후 `z^c` 예측 오차, α의 충돌 탐지 AUROC |
| 시드 | 3개 × 50 에피소드 |

## 일정

| 날짜 | 마일스톤 |
|---|---|
| 10/12 (월) | 프로포절 발표. 충돌 환경 프로토타입, 파일럿 |
| ~10/26 | CF-JEPA 재구현, 환경 ①의 수치 재현 (TwoRoom 방해물 2개에서 ≈0.95) |
| 11/5 (목) | 중간 발표. 환경 ② 완성, 게이트 첫 결과 |
| 11/26 (목) | 최종 발표. 두 환경 결과표, ablation, α 시각화 데모 |

## 팀원

| 이름 | GitHub |
|---|---|
| 강연우 | [@yeonooe](https://github.com/yeonooe) |
| 조하은 | [@gkdms04](https://github.com/gkdms04) |
| 설규원 | [@seolcu](https://github.com/seolcu) |

## 폴더 구조

```
src/        모델 코드 (인코더, 예측기, 게이트, 손실)
envs/       충돌 장애물 환경
configs/    학습·평가 설정
notebooks/  Colab 실험 노트북
scripts/    데이터 수집, 학습, 평가 실행 스크립트
```

데이터와 체크포인트는 커밋하지 않습니다(`.gitignore`). Google Drive나 Hugging Face로 공유하세요.

## 참고 자료

- [LeWM 논문](https://arxiv.org/abs/2603.19312)
- [CF-JEPA 논문](https://arxiv.org/abs/2610.00727) ([HTML](https://arxiv.org/html/2610.00727))
- [SMWM (Sensorimotor World Models)](https://arxiv.org/abs/2606.20104)
- [A Control Theory of Predictability in Latent World Models](https://arxiv.org/abs/2607.10362)
