# AGENTS.md

This file provides guidance to AI agents when working with code in this repository.

연구 개요, 수식, 실험 설계는 `README.md`에 있다. 이 파일에는 README만 봐서는 알 수 없는 맥락, 결정 이유, 작업 규칙을 적는다.

## 프로젝트 맥락

- 아주대학교 2026-2학기 인공지능 수업의 팀 프로젝트다.
- 발표 일정: 프로포절 10/12, 중간 11/5, 최종 11/26.
  - 중간 발표는 5분 안에 프로포절 이후 바뀐 것만 발표한다(10/29 수업 공지).
- 평가 기준 (수업에서 강조된 것):
  - 직접 학습해서 성능이 수치로 개선되는가
  - 최신(2025~26) baseline을 쓰는가
  - 개선 포인트가 module 한두 개로 분명한가
  - 결과를 보여 주는 demo가 있는가
  - prompt engineering, agent, RAG 위주가 아닌가
  - 결과의 성패보다 과정과 노력이 드러나는가
- 연산 자원은 Colab Pro+를 활용한다.

## 발표 피드백 요약

- 기존 benchmark task의 success rate를 함께 제시한다. Collision env는 우리가 만든 환경이므로 Distractor env(기존 task) 결과를 함께 앞세운다.
- 선행연구를 충분히 조사한다.
- Obstacle이 많아질 때(object 간 다중 관계, 조합 수 증가) 모델링할 수 있는지 답한다. 현재 gate `α_t`는 `z^u` 전체에 대한 scalar 하나라 직접 해당되는 질문이다. Collision env의 obstacle 개수를 늘리는 실험으로 답하고, 안 되면 한계와 향후 과제로 정리한다.
- 중간 발표에는 제안 모델(gate)의 결과가 있어야 한다.
- 성능이 꼭 SOTA일 필요는 없고, 과정과 그 과정에서 배운 것을 보여 준다.

## 결정 사항과 이유

- **주제 확정 (10/9):**
  - 근거: CF-JEPA 저자가 Conclusion(§V)에 independence assumption을 limitation으로 직접 적었다.
- **언어 조건 world model:** 범위 밖으로 결정했다.
- **주 환경:** TwoRoom이다.
  - 2D라 collision obstacle을 추가하기 쉽다.
  - CF-JEPA 성능이 높아서(0.95) 차이가 깨끗하게 보인다.
  - PushT는 Distractor env(robustness 유지 확인)에만 쓴다.
- **중간 발표(11/5) 우선순위 (11/1):** 1) CF-JEPA 재구현과 TwoRoom N=2 재현, 2) Collision env 완성, 3) gate 초기 결과, 4) obstacle 개수 늘리기.
  - 4)는 중간 발표 범위에서 빼지 않는다. 시간이 남으면 시도하되 1순위는 아니다.
- **실행 위치:** 코드는 로컬에서 쓰고, 실행(smoke test 포함)은 Colab에서 한다. 로컬에는 Python 3.10이 없다.

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

- **CF-JEPA가 꼽은 가장 가까운 prior:** SMWM(inverse dynamics 쪽), TIA와 Denoised MDP(latent 분리 쪽, 둘 다 reconstruction 필요). Iso-Dream 계열은 인용하지 않았다.
- **환경 정의 (§III-A Eq. 1):** `T(s′|s, a) = T_c(s^c′|s^c, a) · T_u(s^u′|s^u)`, `x_t = q(s^c_t, s^u_t)`이다. 저자는 이 구조를 "exogenous block MDP"라 부르며 Efroni et al.(ICLR 2022)을 인용한다. `T_c`에는 `s^u`가 들어가지 않고, 두 부분이 만나는 곳은 emission `q`뿐이다.
- **Distractor 배치:** task를 가리지 않게 배치한다. 저자는 method 절에서 "task를 직접 가리는 것은 포착할 수 없다"는 한계도 따로 적었다.

### 선행연구에서 확인한 사실 (10/9, 원문 기준)

- **Iso-Dream++ (2303.14889, arXiv 2023):** 가장 가까운 선행연구다. 학술지 게재 정보는 arXiv에도 없다.
  - 저자는 v2 기준 5명(Pan, Zhu, Zheng, Wang, Yang)이다. v1에는 Zheng이 없다.
  - Inverse dynamics로 controllable / noncontrollable branch를 분리한다.
  - Sparse dependency gate의 방향은 controllable → noncontrollable(s_t → z_{t+1})이다. FC + sigmoid를 0.5에서 hard threshold한다.
  - Controllable transition prior는 noncontrollable state를 받지 않는다. Noncontrollable 미래는 policy의 attention 입력으로만 쓰인다.
  - Dreamer 기반이라 reconstruction과 reward가 필요하다. 환경은 CARLA와 DMC video background다.
- **Denoised MDP (ICML 2022):** controllable·reward-relevant x의 transition이 p(x_t | x_{t-1}, a)라 noise latent를 받지 않는다.
- **SPARTAN (2411.06890, NeurIPS 2025):**
  - Object token 간 attention을 Bernoulli mask(Gumbel-softmax)로 만들고, edge 수에 Lagrangian sparsity penalty를 건다.
  - Object latent는 ground-truth mask나 slot에서 얻는다. 예측만 평가하고 planning은 하지 않는다.
- **Causal-JEPA (2602.11389):** object-level latent masking으로 interaction을 암묵적으로 학습한다. Frozen VideoSAUR(DINOv2 기반 slot) encoder를 쓰고, Push-T에서 CEM으로 planning한다.
- **2607.10362:** planner가 도달하는 state에서의 예측 충실도가 control 성공을 좌우하고, data 평균 예측 오차와는 거의 무관하다는 분석이다. Controllability와는 관계가 없다. 평가에서 success rate를 주 지표로 두는 근거로 쓴다.
- **2608.20065 (Orthogonal JEPA):** 저자가 철회했다. 후속 논문 2609.20800(JEPA-Anything)은 범용 factorization이라 인용하지 않는다.

### 구현 자원에서 확인한 사실 (10/10)

- **LeWM 코드:** github.com/lucas-maes/le-wm, MIT. 환경, planning, 평가는 stable-worldmodel을, 학습은 stable-pretraining을 쓴다. 설치는 `uv pip install stable-worldmodel[train,env]`(Python 3.10)이다. README의 예시 명령은 PushT 것뿐이다(`python train.py data=pusht`).
- **TwoRoom data:** Hugging Face dataset `quentinll/lewm-tworooms`(MIT, gated 아님)다. `tworoom.tar.zst` 약 3.4GB를 받아 `$STABLEWM_HOME`(기본 `~/.stable-wm/`)에 `.h5`로 푼다.
- **TwoRoom checkpoint:** Hugging Face model `quentinll/lewm-tworooms`(MIT, gated 아님)다. `weights.pt`와 `config.json`, 약 145MB다.
- **stable-worldmodel:** github.com/galilai-group/stable-worldmodel, MIT. 10/2까지 갱신 중이라 API가 바뀔 수 있다.
  - TwoRoom은 `stable_worldmodel/envs/two_room/env.py`에 있고, 환경 ID는 `swm/TwoRoom-v1`이다. 데이터 수집 script는 `scripts/data/collect_tworooms.py`다.
  - 순수 PyTorch로 렌더링과 충돌을 처리한다(224×224, action은 `Box(-1, 1, (2,))`의 속도 방향).
  - `step()`은 `pos + action * speed`를 구한 뒤 `_apply_collisions()`로 벽과 테두리를 clamp한다.
  - 따라서 움직이는 obstacle은 subclass로 넣을 수 있다. obstacle 상태를 추가하고, `step()`에서 이동과 push를 처리하고, `_render_frame()`에 그리면 된다.
  - 새 obstacle은 variation space에 직접 등록하지 않으면 factor of variation으로 다룰 수 없다.

### 실행 환경에서 확인한 사실 (10/11)

- **실행:** Colab Pro+는 Google 공식 Colab CLI(`google-colab-cli`)로 쓴다. notebook 파일 없이 runtime만 빌려 터미널에서 실행한다. 학교 계정으로 로그인한다.
- **LeWM TwoRoom 재현:** success 0.86(50 episodes, T4)이다. 논문 값은 0.87이다. 절차와 수치는 `docs/experiments.md` E0에 있다.
- **설치 함정:** 모두 `scripts/colab/setup_lewm.sh`에 반영했다.
  - Python은 3.10 venv를 쓴다.
  - stable-worldmodel은 `[train,env,format]` extra로 설치한다.
  - `datasets>=3`이 필요하다.
  - `MPLBACKEND=Agg`를 설정한다.
  - dataset은 `$STABLEWM_HOME/datasets/`에 둔다.
- **Checkpoint 호환성:** HF `weights.pt`는 transformers v4의 ViT key 이름이고, stable-pretraining은 v5를 요구한다. `scripts/colab/fix_lewm_keys.py`로 key 이름만 바꾸면 strict load가 된다. stable-worldmodel 0.1.1의 `load_pretrained`는 HF repo 이름을 직접 받는다.
- **Collision env prototype:** `envs/collision_tworoom.py`에 있다(E1). 원래 TwoRoom은 goal을 화면에 그리지 않고 agent를 Gaussian blob(σ = radius)으로 그린다.

### 아직 확인하지 않은 것

- CF-JEPA가 distractor를 stable-worldmodel에 어떻게 넣었는지(코드 미공개)
- Iso-Dream++의 학술지 게재본(TPAMI 등) 여부

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
- **환경 이름 (11/1 정함):** 실험 환경은 영어 이름으로만 부른다. 충돌하지 않는 사각형이 있는 환경은 **Distractor env**(TwoRoom이면 Distractor TwoRoom), 움직이는 obstacle이 agent를 미는 환경은 **Collision env**(Collision TwoRoom)다. 예전 표기(Env 뒤에 동그라미 숫자)와 한국어 이름(사각형 환경, 충돌 환경)은 쓰지 않는다.

## 협업 규칙 (11/1 정함)

- **역할 (중간 발표까지):** 강연우는 gate 구현과 모델 학습, 설규원은 Collision env와 data, 조하은은 평가와 발표 준비를 맡는다. CF-JEPA 재구현은 설규원이 했다(E3).
- **코드 (GitHub):** 팀원 3명 모두 collaborator다. 작업 단위로 branch(`<이름>/<작업>`, 예: `yeonwoo/gate`)를 만들고 PR로 main에 합친다.
- **결과물 (Proton Drive):** 그림, 영상, checkpoint, data는 공유 폴더의 `results/E<번호>_<이름>/`에 올리고, 폴더의 `README.txt`에 한 줄 설명을 더한다.
- **실험 기록:** `docs/experiments.md`에 E 번호를 이어서 붙인다(현재 E9까지). 목적, 설정, 실행 명령, 결과, 남은 문제를 적는다. 실험 번호를 먼저 정하고 쓰면 번호가 겹치지 않는다.
- **Colab:** 각자 자기 계정을 쓴다. 환경 준비는 `scripts/colab/setup_lewm.sh` 하나로 된다(약 10분, data 포함).
- **팀원에게 일을 줄 때:** 구체적인 구현 방법보다 요구사항(무엇이 되어야 하는지, 어떻게 확인하는지)을 준다. 간단한 일은 설규원이 직접 한다.

## 외부 자료

- **팀 공유 폴더 (Proton Drive):** `/my-files/문서/interaction-aware-wm` (설규원 계정, 링크 공유)
  - git에 넣기 애매한 파일(영상, 그림, 결과 파일, 발표 자료)은 여기에 둔다. 코드와 실험 일지는 GitHub에 둔다.
  - 현재 내용: `README.txt`(폴더 안내), `results/E0_lewm_tworoom_baseline`, `results/E1_collision_tworoom`, `results/E2_oracle`, `인공지능_17팀_발표자료.pptx`.
  - 로컬에 Proton Drive CLI(`proton-drive`, `~/.local/bin`)가 있으면 그것을 쓴다.
    - 목록: `proton-drive filesystem list <path>`
    - 내려받기: `proton-drive filesystem download <path>... <localFolder>`
    - 올리기: `proton-drive filesystem upload <localPath>... <parentPath>`
- **프로포절 발표 자료:** 공유 폴더의 `인공지능_17팀_발표자료.pptx` (팀원 제작, 10/12 발표에 사용, 9장)
  - 구성: 표지 / Topic / Motivation / Objectives / Background(CF-JEPA, TwoRoom 0.87·0.95 / 0.33·0.95, §V 인용) / Related Works / Proposed Method / Baseline Reproduction(LeWM 0.86) / Collision TwoRoom(seed 0~3 frame, "추후: 충돌·비충돌 구간의 예측 오차와 α_t 비교").
  - 슬라이드 글자가 모두 이미지라 python-pptx로는 텍스트가 나오지 않는다. `soffice --headless --convert-to pdf`로 바꿔서 본다.
  - E2 oracle 실험은 분량 때문에 일부러 넣지 않았다. 중간 발표에서 활용한다.
- **팀원 GitHub:** 강연우 yeonooe, 조하은 gkdms04, 설규원 seolcu.
