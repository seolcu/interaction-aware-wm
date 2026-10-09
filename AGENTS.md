# AGENTS.md

This file provides guidance to AI agents when working with code in this repository.

연구 개요, 수식, 실험 설계는 `README.md`에 있다. 이 파일에는 README만 봐서는 알 수 없는 맥락, 결정 이유, 작업 규칙을 적는다.

## 프로젝트 맥락

- 아주대학교 2026-2학기 인공지능 수업의 팀 프로젝트다.
- 발표 일정: 프로포절 10/12, 중간 11/5, 최종 11/26.
- 평가 기준 (수업에서 강조된 것):
  - 직접 학습해서 성능이 수치로 개선되는가
  - 최신(2025~26) baseline을 쓰는가
  - 개선 포인트가 module 한두 개로 분명한가
  - 결과를 보여 주는 demo가 있는가
  - prompt engineering, agent, RAG 위주가 아닌가
  - 결과의 성패보다 과정과 노력이 드러나는가
- 연산 자원은 Colab Pro+를 활용한다.

## 결정 사항과 이유

- **주제 확정 (10/9):**
  - 근거: CF-JEPA 저자가 Conclusion(§V)에 independence assumption을 limitation으로 직접 적었다.
- **언어 조건 world model:** 범위 밖으로 결정했다.
- **주 환경:** TwoRoom이다.
  - 2D라 collision obstacle을 추가하기 쉽다.
  - CF-JEPA 성능이 높아서(0.95) 차이가 깨끗하게 보인다.
  - PushT는 Env. ①(robustness 유지 확인)에만 쓴다.
- **전환 계획:** 10/26까지 CF-JEPA 재구현이 Env. ① 수치를 재현하지 못하면, "LeWM + gate" 단순화 버전으로 전환한다. 이때 환경은 그대로 재사용한다.

## 원문에서 확인한 사실

### CF-JEPA (arXiv 2610.00727)

- **코드:** 공개되지 않았다. 논문에는 stable-worldmodel과 stable-pretraining을 고쳐 썼다고만 적혀 있다.
- **Latent:** 192-dim을 z^c 128 + z^u 64로 나눈다.
  - g는 AdaLN-zero로 action을 받는 transformer다.
  - h는 action을 받지 않는 작은 transformer다.
- **Loss:** L = L_fwd + α·L_inv + β·L_adv + γ·L_SIGReg
  - β = 1, γ = 0.09
  - α는 task별로 다르다: Reacher 10, PushT 3, OGBench Cube 10, TwoRoom 3.
- **Planning:** CEM, 후보 300개, 반복 30회, 5 step 계획 후 receding horizon 5. Cost는 z^c distance만 쓴다.
- **Training:** AdamW, lr 5e-5, weight decay 1e-3, 10 epochs, batch 128. RTX 5090 1장에서 모델당 8시간 미만.
- **Distractor:**
  - 단일 색 사각형이 이미지 가장자리에서 등장한다.
  - 크기: 길이 49px, 폭 18~36px.
  - 속도: 3~18px/step.
  - 기본 개수는 N = 2다.
- **Data:** TwoRoom은 노이즈 섞인 scripted policy로 모은 1만 episode, 92만 frame이다.
- **결과 (success rate):**

  | Task | 조건 | LeWM | SMWM | CF-JEPA |
  |---|---|---|---|---|
  | Reacher | distractor 없음 (Table II) | 0.86 | 0.66 | 0.61 |
  | PushT | distractor 없음 (Table II) | 0.96 | 0.83 | 0.78 |
  | OGBench Cube | distractor 없음 (Table II) | 0.74 | 0.84 | 0.81 |
  | TwoRoom | distractor 없음 (Table II) | 0.87 | 0.99 | 0.95 |
  | Reacher | N = 2 (Table III) | 0.07 | 0.10 | 0.48 |
  | PushT | N = 2 (Table III) | 0.01 | 0.86 | 0.87 |
  | OGBench Cube | N = 2 (Table III) | 0.42 | 0.84 | 0.85 |
  | TwoRoom | N = 2 (Table III) | 0.33 | 0.98 | 0.95 |

  - Table IV에는 LeWM 수치가 조금 다르게 나온다(Reacher 0.08, TwoRoom 0.34). 우리는 Table III 값을 인용한다.
- **원문 인용 (§V):**
  - "we assume that the uncontrollable part of the scene never affects the controllable part"
  - "a person working next to the robot, would require extending the model"

### 아직 확인하지 않은 것

- LeWM TwoRoom checkpoint와 data를 공개적으로 받을 수 있는지
- stable-worldmodel의 TwoRoom 환경에 object를 추가하는 난이도
- 2607.10362(Control Theory of Predictability), 2608.20065(Orthogonal JEPA), 2602.11389(Causal-JEPA)는 초록만 확인했다

## 작업 규칙

- **문체:**
  - 모든 문서는 연구 제안서 톤으로, 한국어 "~이다" 체로 쓴다.
  - 전문 용어는 영어 원문 그대로 쓴다.
- **README에 넣지 않는 것:** 역할 분담, 수업 운영 정보(팀 번호, 발표 그룹).
- **커밋하지 않는 것:** dataset, checkpoint, wandb 로그.
- **기호와 코드 이름:**
  - 기호 z^c, z^u, α_t, g, h, h_int는 README와 같은 뜻으로 쓴다.
  - 코드 변수명도 이를 따른다: `z_c`, `z_u`, `alpha`, `h_int`.
- **커밋 메시지:** 영어로 쓴다.

## 외부 자료

- **프로포절 덱 초안:** https://claude.ai/artifact/QNbenSN6xX6ZmDn74ggiKn
  - 9장이고, 발표자 노트에 예상 질문과 출처가 있다.
  - 설규원 계정의 Claude artifact다.
  - 실제 발표에 사용할 덱은 따로 제작한다.
- **팀원 GitHub:** 강연우 yeonooe, 조하은 gkdms04, 설규원 seolcu.
