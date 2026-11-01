# Experiment Log

실험마다 날짜, 목적, 실행 방법, 결과, 남은 문제를 적는다. dataset, checkpoint, 영상은 git에 넣지 않는다.

## 2026-10-11 · E0. LeWM TwoRoom baseline 재현

**목적:** 공개 LeWM TwoRoom checkpoint가 우리 Colab 환경에서 논문 수치를 재현하는지 확인한다. 이후 CF-JEPA 재구현과 Collision env의 출발점이다.

**환경**
- Colab Pro+, Tesla T4 1장. Google Colab CLI(`google-colab-cli`)로 실행했다.
- Python 3.10 venv, stable-worldmodel 0.1.1, stable-pretraining 0.1.7, transformers v5.
- le-wm commit `8edfeb3` (2026-05-22).

**실행**

```bash
colab new -s lewm --gpu T4
colab upload -s lewm scripts/colab/<file> /content/iawm/scripts/colab/<file>   # setup_lewm.sh, fix_lewm_keys.py, eval_lewm_tworoom.sh
# VM 안에서
bash /content/iawm/scripts/colab/setup_lewm.sh          # 약 10분, data 3.4GB 압축 / 12.8GB 해제
bash /content/iawm/scripts/colab/eval_lewm_tworoom.sh   # 약 3분
```

**설정:** `config/eval/tworoom.yaml` 그대로 사용했다.
- CEM: 300 samples, 30 iterations, top-k 30, horizon 5, receding horizon 5, action block 5.
- 평가: 50 episodes, goal offset 25 steps, eval budget 50, seed 42.

**결과**

| | Success rate |
|---|---|
| CF-JEPA Table II의 LeWM 값 (distractor 없음) | 0.87 |
| **우리 재현** | **0.86** (43/50) |

- 평가 시간: 155초 (T4).
- 영상 예시: `outputs/lewm_tworoom_baseline/env_0.mp4`(성공), `env_1.mp4`(실패). 로컬에만 있다.

**설치 중 만난 문제와 해결** (script에 반영함)
1. Colab 기본 Python 3.13에는 box2d-py와 labmaze wheel이 없다 → uv로 Python 3.10 venv를 만든다.
2. resolver가 `datasets==1.1.1`을 골라 pyarrow 최신과 충돌한다 → `datasets>=3`.
3. Colab kernel의 `MPLBACKEND`가 venv로 상속되어 matplotlib import가 실패한다 → `MPLBACKEND=Agg`.
4. HF `weights.pt`는 transformers v4 ViT key 이름이고 stable-pretraining은 v5를 요구한다 → `fix_lewm_keys.py`로 key 이름만 바꾼다. strict load로 검증했다.
5. HDF5 지원은 `format` extra에 있다 → `stable-worldmodel[train,env,format]`.
6. stable-worldmodel 0.1.x는 dataset을 `$STABLEWM_HOME/datasets/`에서 찾는다.

**실행 기록:** `notebooks/logs/2026-10-11_e0_lewm_baseline_session.ipynb`는 Colab CLI 세션에서 실행한 모든 cell과 출력을 그대로 내보낸 것이다(`colab log -o`). 시행착오도 함께 들어 있다.

**다음:** 같은 pipeline에 Collision env obstacle을 넣는다(B). CF-JEPA 재구현의 기준선으로 이 0.86을 쓴다.

## 2026-10-11 · E1. Collision TwoRoom prototype

**목적:** CF-JEPA의 independence assumption이 깨지는 상황, 즉 uncontrollable obstacle이 agent를 미는 환경을 만든다.

**구현:** `envs/collision_tworoom.py`의 `CollisionTwoRoomEnv`는 stable-worldmodel `TwoRoomEnv`의 subclass다.
- **Obstacle 설정:** 원형 1개, 반지름 14px(agent의 2배), 속도 3~6 px/step이다. 테두리와 중앙 벽에서 반사되어 자기 방에 머문다.
- **Obstacle은 action과 agent의 영향을 받지 않는다.** 영향은 u → c 한 방향뿐이다.
- **매 step 처리 순서:**
  1. agent가 이동한다.
  2. obstacle이 이동한다.
  3. 겹치면 agent를 중심선 방향으로 밀어낸다.
  4. 부모의 벽 처리를 한다.
  - 벽 처리가 agent를 다시 obstacle 안으로 넣을 수 있어 3~4를 최대 3번 반복한다. 그래도 겹치면 `pinned`로 표시한다.
- **`info` 추가 항목:** `obstacle_pos`, `obstacle_vel`, `contact`, `pinned`.
- **Spawn (`crossing`):** agent → target 경로를 가로지르도록 obstacle을 배치한다.
- **Goal image:** obstacle을 그리지 않는다.

**확인** (`scripts/colab/demo_collision.py`, Colab CPU runtime, 약 16초)
- `num_obstacles=0`이면 같은 seed와 action에서 궤적이 부모 `TwoRoomEnv`와 정확히 같다(200 steps).
- 200 episodes 동안 매 step 두 조건을 assert했고 모두 통과했다.
  - pinned가 아니면 agent와 obstacle이 겹치지 않는다.
  - push 때문에 agent가 벽 안으로 들어가지 않는다.
- 부모 TwoRoom 자체에서도 문을 지날 때 agent 중심이 벽 가장자리에 걸치는 일이 있다. obstacle 없이 같은 policy로 200 episode 중 10개였다. 그래서 벽 검사는 contact step에만 적용했다.

**Contact 통계** (`crossing`, goal 직진 + noise scripted policy, 200 episodes)

| 지표 | 값 |
|---|---|
| contact가 한 번 이상 있는 episode | 67.5% |
| contact step 비율 | 13.7% |
| pinned step 비율 | 0.9% |
| goal 도달 (100 step 안) | 100% |
| 평균 episode 길이 | 28.8 steps |

- Contact step 비율은 처음 기대(10% 미만)보다 조금 높다. scripted policy가 obstacle을 피하지 않고 계속 goal 쪽으로 밀고 들어가서 contact가 몇 step씩 이어지기 때문이다.
- 그래도 대부분의 step은 non-contact이다. 실제 data 수집 설정(spawn, policy)은 C와 본 실험에서 정한다.

**결과물:** `outputs/collision_tworoom/demo_seed{0..3}.gif`와 `*_sheet.png`(contact 전후 프레임)다. 로컬에만 있다. Contact 프레임은 주황 테두리, goal 위치는 초록 X로 표시했다. 둘 다 GIF에만 덧그린 시각화이고, env observation에는 goal을 그리지 않는다(TwoRoom 기본값).

**아직 하지 않은 것:** gym ID 등록, variation space 등록, `World`와 `eval.py` 연동, data 수집.

## 2026-10-11 · E2. Oracle 실험: z^u를 보지 않는 predictor의 collision 오차

**목적:** 7번 슬라이드의 주장을 실험으로 확인한다. 주장은 "g의 입력에 z^u가 없으면 collision 순간에 줄일 수 없는 오차가 남는다"는 것이다.
- Pixel과 encoder는 쓰지 않는다. ground-truth state로 입력 정보 하나만 바꿔 비교한다.

**설정** (`scripts/colab/oracle_irreducible.py`, T4, 약 7분)
- **Data:** E1 환경에서 scripted policy(goal 직진, noise 0.5)로 모았다.
  - 학습용은 3000 episodes, 약 9.8만 transition이다.
  - 평가용은 따로 600 episodes를 모았고, contact step 비율은 13.7%다.
  - obstacle 없는 대조군도 같은 방식으로 모았다.
- **예측 대상:** 다음 step의 agent 이동량(Δpos, px)이다.
- **모델:** 3-layer MLP. Adam lr 1e-3에 cosine schedule을 쓰고, 60 epochs 학습했다. seed는 3개다.
  - `g`: 입력은 (agent pos, action)이다. CF-JEPA의 g(z^c, a)에 해당한다.
  - `g + u`: 입력은 (agent pos, obstacle pos·vel, action)이다. u → c 경로가 있는 predictor다.

**결과** (평가용 data의 RMSE, px, 3 seeds 평균±표준편차)

| 환경 | 모델 | non-contact | contact |
|---|---|---|---|
| obstacle 없음 | g (h=256) | 0.44 ± 0.03 | - |
| collision | g (h=256) | 0.94 ± 0.01 | **4.71 ± 0.00** |
| collision | g (h=1024) | 0.72 ± 0.01 | **4.69 ± 0.00** |
| collision | g + u (h=256) | 0.86 ± 0.00 | **1.75 ± 0.04** |

**학습 곡선** (마지막 seed, contact RMSE, epoch 12 → 60)
- g (h=256): 4.70 → 4.71. 줄지 않는다.
- g (h=1024): 4.67 → 4.69. 줄지 않는다. 같은 기간 non-contact는 1.06에서 0.70으로 줄었다.
- g + u (h=256): 3.77 → 1.70. 계속 줄었다.

**해석**
- g의 contact 오차는 약 4.7px다. agent가 한 step에 움직이는 거리(약 5px)와 비슷하다. 즉 부딪히는 순간 g는 한 걸음만큼 틀린다.
- **capacity를 4배로 늘려도 contact 오차는 그대로였다.** 학습을 오래 해도 epoch 12부터 평평하다. 반면 non-contact 오차는 계속 줄었다. capacity와 학습 시간으로 줄지 않는 오차다. irreducible error 주장과 맞는다.
- **obstacle 정보를 입력에 넣으면 contact 오차가 63% 줄었다(4.71 → 1.75).** 그러니 부족한 것은 입력 정보다.
- g + u도 0까지는 내려가지 않았다. push와 벽 처리 때문에 경계가 불연속인데, 작은 MLP가 이를 근사하는 한계로 본다. 확인하지는 않았다.

**한계:** ground-truth 좌표를 쓴 oracle 실험이다. 학습된 latent(z^c, z^u)에서도 같은 차이가 나는지는 본 실험에서 확인한다.

**참고:** 같은 script를 CPU runtime에서 먼저 돌렸지만, h=1024 학습이 45분 넘게 끝나지 않아 중단했다. 학습은 GPU에서 한다.

## 2026-11-01 · E3. CF-JEPA 재구현과 Distractor TwoRoom (N=2)

**목적:** 교수님 요청대로 기존 task(TwoRoom)에서 CF-JEPA 수치를 재현한다. 목표는 CF-JEPA Table III의 0.95(N=2)다. 이후 gate 실험의 baseline이 된다.

**구현** (공식 코드가 없어서 논문 §III-C, §III-D를 따라 다시 만들었다)
- `src/iawm/cfjepa.py`: stable-worldmodel 0.1.1의 `LeWM`을 subclass한다. `rollout`과 `get_cost`를 물려받으므로 E0와 같은 CEM 평가 pipeline을 쓴다.
  - z(192)를 z^c 128과 z^u 64로 나눈다.
  - `g`: AdaLN-zero action 조건 transformer(LeWM predictor와 같은 크기, depth 6)다.
  - `h`: action 없는 causal transformer다.
  - ψ(inverse), φ(gradient reversal + adversarial)는 2-layer GELU MLP다.
  - Loss = L_fwd + 3·L_inv + 1·L_adv + 0.09·SIGReg(TwoRoom 값)다.
  - Planning cost는 z^c만 쓰고, Eq. 7처럼 매 step의 L2 거리를 더한다.
- `src/iawm/distractors.py`: 논문 §IV-C-1의 사각형이다. 가장자리에 붙어 있고 길이 49px, 폭 18~36px, 속도 3~18 px/step이다. 논문에 "학습 이미지에 N=2 distractor를 augment했다"고 적혀 있어서, data를 다시 모으지 않고 학습 batch에 GPU에서 덧그린다. Dataset이 frameskip 5라서 frame 사이에 5 step만큼 움직인다.
- `envs/distractor_tworoom.py`: 평가용이다. `render()`가 돌려주는 frame에만 사각형을 그린다. 충돌하지 않고 dynamics는 원래 TwoRoom과 같다. gym ID는 `iawm/TwoRoomDistractor-v1`이다.
- `scripts/train.py`, `scripts/eval.py`: le-wm의 train.py와 eval.py를 고쳐 썼다. `model=lewm`으로 바꾸면 같은 조건의 LeWM baseline도 학습할 수 있다.

**논문에 없어서 우리가 정한 것** (11/1 팀 합의)
- Goal image에는 사각형을 그리지 않는다. 평가 pipeline이 dataset frame을 goal로 쓰는 기본 동작을 그대로 둔다. CF-JEPA는 z^c만 비교하므로 영향이 작다고 본다.
- 사각형이 화면 끝에 닿으면 튕겨 돌아온다. 그래서 항상 N개가 모두 보인다. 색은 RGB 균일 분포에서 고른다.
- `h`는 g의 절반 정도 크기다(depth 3, heads 8, hidden 128).
- LeWM의 pred_proj(BatchNorm MLP)는 z^c와 z^u를 섞으므로, z^c용과 z^u용 pred_proj를 따로 둔다.
- ψ, φ의 hidden은 512다.

**Smoke test** (`scripts/colab/smoke_cfjepa.py`, A100, 모두 통과)
- 사각형 면적이 항상 49×폭이다. 즉 화면 밖으로 나가지 않는다. 정규화 공간에서 칠한 색을 되돌리면 원래 RGB와 같다.
- Gradient reversal이 backward에서 부호를 뒤집는다.
- z^u를 바꿔도 z^c 예측은 그대로다. action을 바꿔도 z^u 예측은 그대로다.
- Loss 4개가 유한하고, gradient가 encoder와 φ까지 간다.
- `get_cost`는 (B, S) cost를 돌려주고 z^c만 본다.
- `num_distractors=0`이면 원래 TwoRoom과 같은 frame이 나온다.
- 300 step 학습 후 5 episode 평가가 끝까지 돈다(pipeline 확인용이라 수치는 의미 없음).
- 그림: `outputs/cfjepa/distractor_env_sheet.png`(env에서 연속한 20 step, 한 줄에 5 step). 로컬과 Proton Drive에 있다.

**본 학습** (11/1 05:00~09:50 KST)
- Colab A100 40GB, 1 seed(seed 3072), 10 epochs, batch 128, bf16-mixed이다.
- 1 epoch는 5,138 step이고, 속도는 약 3.4 it/s다. 10 epoch에 약 4시간 50분 걸렸다.
- 명령: `bash scripts/colab/train_cfjepa.sh output_model_name=cfjepa_tworoom_n2`
- 학습 loss는 spt가 `/root/.cache/stable-pretraining/runs/<날짜>/<시각>/<id>/metrics.csv`에 `fit/*`로 남긴다. CSVLogger 경로가 아니다.
- Validation loss (epoch 1 → 10): fwd 0.048 → 0.013, inv 0.770 → 0.753, SIGReg 5.0 → 1.48, adv는 내내 약 1.0이다.
  - adv ≈ 1은 φ가 z^u에서 action을 맞히지 못한다는 뜻이다(action 분산과 같은 수준의 오차). 즉 z^u에 action 정보가 남지 않았다. 의도한 결과다.

**결과** (CEM 300×30, 50 episodes, eval seed 42, `scripts/colab/eval_cfjepa.sh`)

| 모델 | 학습 data | 평가 N=0 | 평가 N=2 |
|---|---|---|---|
| CF-JEPA (우리 재구현) | N=2 augment | **0.94** (논문 0.95) | **0.88** (논문 0.95) |
| LeWM (HF checkpoint, E0) | 깨끗한 data | 0.86 (논문 0.87) | 0.84 |

- CF-JEPA N=0은 논문과 거의 같다. N=2는 0.07 낮다. 50 episode의 표준오차가 약 0.05이고 우리는 seed 1개(논문은 3개 평균)라서, 논문 범위 안이라고 볼 수 있다. 재구현이 동작한다고 판단한다.
- 실패 6개 중 무엇이 원인인지(distractor인지 원래 어려운 시작점인지)는 아직 보지 않았다. 같은 50개 시작점에서 N=0은 3개 실패다.
- **LeWM N=2는 논문 수치(0.33)와 비교할 수 없다.** 논문은 "All models were trained with images augmented with N=2 distractors"라고 적었다. 즉 논문의 LeWM 0.33은 distractor로 학습한 LeWM이다. 우리 0.84는 깨끗한 data로 학습한 공개 checkpoint를 distractor 환경에서 평가한 것이다. 논문 조건의 LeWM baseline을 얻으려면 `model=lewm`, distractor N=2로 다시 학습해야 한다(A100 약 4~5시간).
  - 깨끗하게 학습한 LeWM은 distractor에 거의 영향을 받지 않는다는 것 자체도 기록할 만하다. 논문의 LeWM 붕괴는 "보지 못한 사각형"이 아니라 "학습 중 latent가 사각형을 담게 된 것" 때문이라는 해석과 맞는다.
- 그림 확인: 평가 영상의 agent 화면에 사각형 2개가 보이고 goal 화면에는 없다(설계대로).

**결과물**
- Checkpoint: Proton Drive `results/E3_cfjepa_tworoom/weights_epoch_10.pt`, `config.json`. 다른 epoch는 설규원 로컬 `checkpoints/cfjepa_tworoom_n2/`에 있다.
- 평가 결과와 영상(episode별 mp4, agent / dataset / goal 3분할): Proton Drive `results/E3_cfjepa_tworoom/eval/`, 로컬 `outputs/cfjepa/eval/`.

**남은 문제**
- 논문 조건의 LeWM baseline(N=2로 학습)이 없다. 중간 발표 표에서 "LeWM 0.33"은 논문 인용으로 표시해야 한다.
- Seed 1개다. 최종 발표 전 3 seed로 늘린다(11/1 결정).

## 2026-11-01 · E4. Collision TwoRoom 학습 data

**목적:** Collision TwoRoom에서 CF-JEPA와 gate를 학습할 data를 만든다. 원래 TwoRoom data와 조건을 맞춰서 Distractor TwoRoom 결과와 비교할 수 있게 한다.

**원래 TwoRoom data** (`tworoom.h5`, Colab에서 직접 열어 확인)
- 10,000 episode, 920,809 frame, episode당 평균 92 step이다.
- Column: `pixels`(224×224×3), `action`(2), `proprio`·`pos_agent`(agent 위치), `pos_target`, `ep_idx`, `step_idx`, `terminated` 등이다.
- 모델 학습에는 `pixels`와 `action`만 쓴다. `proprio`는 평가 시작 상태를 정하는 데 쓴다.
- pixels는 blosc(lz4)로 압축되어 있다. 원래 138.6GB인데 저장 크기는 7.9GB다. 그래서 읽을 때 `hdf5plugin` import가 필요하다.
- stable-worldmodel `scripts/data/collect_tworooms.py`로 모았다. 정책은 `ExpertPolicy(action_noise=2.0, action_repeat_prob=0.05)`, episode 최대 100 step, seed 3072다.

**바꾼 것** (`scripts/collect_collision.py`)
- 환경만 `iawm/CollisionTwoRoom-v1`(obstacle 1개, `crossing` spawn)로 바꾼다. 정책, 최대 길이, 해상도, seed는 원래와 같다.
  - E1의 "goal 직진" 정책 대신 원래 정책을 쓴다. 그래서 충돌 통계가 E1과 다르다.
- 추가 column: `obstacle_pos`, `obstacle_vel`, `contact`, `pinned`. 학습에는 쓰지 않고 분석의 정답으로만 쓴다.
- World.collect는 압축 없이 저장하므로(frame당 150KB), 수집 후 원래 data와 같은 blosc 설정으로 다시 쓰고 `ep_idx`를 채운다.
- Collision env는 gym ID `iawm/CollisionTwoRoom-v1`로 등록했다. 평가 때는 World가 시작 위치와 목표를 dataset 값으로 바꾼 뒤(`_set_goal_state`) obstacle을 다시 놓는다. 그래서 평가에서도 obstacle이 시작→목표 경로를 가로지른다.

**확인** (Colab CPU high-mem)
- 같은 seed로 20 episode를 두 번 모으면 `action`, `pixels`, `contact`, `obstacle_pos`가 완전히 같다. 그래서 data 파일(수 GB) 대신 script와 seed를 공유한다. 팀원은 `python scripts/collect_collision.py`로 같은 data를 만들 수 있다.
- 200 episode 시험 수집 결과 (61초, 압축 후 0.11GB):

| 지표 | 값 |
|---|---|
| frame | 12,231 (episode당 평균 61 step) |
| contact step 비율 | 7.7% |
| pinned step 비율 | 0.5% |
| contact가 한 번 이상 있는 episode | 71% |
| goal 도달 episode (100 step 안) | 74.5% |

- 이 data로 `scripts/train.py`(CPU, 3 step)와 `scripts/eval.py --config-name=collision_tworoom`(random policy, 5 episode)이 끝까지 돈다.

**본 수집 결과** (Colab CPU high-mem, 63분, `collision_tworoom.h5` 압축 후 6.10GB)

| 지표 | Collision TwoRoom | 원래 TwoRoom |
|---|---|---|
| episode | 10,000 | 10,000 |
| frame | 659,451 | 920,809 |
| episode당 평균 step | 65.9 | 92.1 |
| contact step 비율 | 7.3% | - |
| pinned step 비율 | 0.5% | - |
| contact가 한 번 이상 있는 episode | 73.5% | - |
| goal 도달 episode | 68.4% | 40.4% |

- Contact는 대부분 episode에서 한 번 이상 일어나고, step 단위로는 7% 정도다. Gate는 "드물게 켜지는" 신호를 배우게 된다.
- Episode가 원래보다 짧다(66 vs 92 step). 원래 data는 62%의 episode가 길이 101(시간 초과)이고 goal 도달은 40%뿐이다. 우리 data는 goal에 더 자주 도달해서 일찍 끝난다. Obstacle이 있는데도 더 자주 도달하므로 obstacle 때문은 아니다. 원래 data가 다른 env 버전(stable-worldmodel에 `legacy_env.py`가 따로 있다)이나 다른 설정으로 모였을 가능성이 있지만 확인하지 않았다. Frame 수가 원래의 72%라 학습 1 epoch도 그만큼 짧아진다.
- 공유: data 파일은 올리지 않는다. 같은 data가 필요하면 `python scripts/collect_collision.py`를 실행한다(seed 3072, CPU로 약 1시간).
- 다음: 이 data로 CF-JEPA와 gate를 학습한다(강연우). 평가는 `bash scripts/colab/eval_cfjepa.sh`가 아니라 `python scripts/eval.py --config-name=collision_tworoom policy=<checkpoint>`로 한다.

## 2026-11-01 · E5. Interaction gate 첫 구현과 학습: collapse, 포화, probe

**목적:** CF-JEPA에 interaction gate를 넣어 Collision TwoRoom data(E4)로 학습한다. Gate가 contact 순간에만 열리는지(RQ2) 확인한다.

**구현** (`src/iawm/cfjepa_gate.py`, `src/iawm/cfjepa.py`는 그대로 두고 복사한 뒤 끝에 추가했다)
- `InteractionGatedPredictor`(`FactorizedPredictor` subclass): `ẑ^c = proj_c(g(z^c, a)) + α · h_int(z^c, z^u, a)`.
  - `h_int`: [z^c, z^u]를 128로 사영한 뒤 action AdaLN을 받는 작은 causal transformer(depth 2, heads 4)다. 출력을 `int_scale · tanh(·)`로 bound하고, 마지막 층을 0으로 초기화해 학습 시작 시 출력이 CF-JEPA와 같다.
  - Gate: `α = σ(MLP([z^c, z^u, Δz^u, a]))`. 한 frame의 z^u로는 obstacle이 다가오는지 알 수 없어 Δz^u를 더했다.
  - `gate_mode`: `learned`(제안), `always`(ablation α=1), `off`(CF-JEPA와 같음). `zu_stopgrad`: ablation.
  - forward 시그니처가 같아 `LeWM.rollout`, CEM 평가가 그대로 동작한다.
- `scripts/train.py`: `[interaction gate]` 주석 블록에서 `ρ·E[α]`를 더하고 α 통계(`gate_alpha_mean/max`, `gate_open_ratio`, `gate_rho`)를 로깅한다. gate가 없는 모델은 이 블록을 건너뛴다.
- Config: `configs/train/model/cfjepa_gate.yaml`, `configs/train/data/collision_tworoom.yaml`(학습 기본값 `tworoom.h5` 대신 `collision_tworoom.h5`를 읽는다).
- 검증: `scripts/colab/smoke_gate.py`(CPU, 1~2분). `off`와 학습 시작 시 `learned`의 출력이 CF-JEPA와 같은지, z^u가 gate를 통해서만 z^c 예측에 들어가는지, stop-gradient, gradient 흐름, `get_cost`를 확인한다. 모두 통과했다.

**학습 설정** (공통): Colab A100 40GB, seed 3072, batch 128, bf16-mixed, Distractor N=2 augment, `data=collision_tworoom`. 1 epoch는 약 3,300 step, 2.9 it/s다.

**E5a. 첫 정식 학습** (`gate_collision_n2`, 10 epoch, 3시간 31분)
- 설정: ρ = 1e-3, 시작 α = σ(0) = 0.5, warmup 없음.
- 명령: `bash scripts/colab/train_cfjepa.sh model=cfjepa_gate data=collision_tworoom output_model_name=gate_collision_n2`
- 결과 (validation, epoch 1 → 10): fwd 0.078 → 0.020, inv 0.82 → 0.77, adv ≈ 1.0. **α 평균 0.0006 → 0.0000, 최대값도 0.0000이다. Gate가 첫 epoch 안에 완전히 닫혔다(collapse).** 결과적으로 h_int가 꺼진 CF-JEPA와 같다.
- 300 step 시험 학습에서는 α가 0.5 → 0.47로 천천히 내려가 문제가 없어 보였다. 300 step(전체의 1%)으로 판단한 것이 잘못이었다.

**E5b. 진단 학습** (`gate_diag`, 1 epoch, 약 40분, 비교용 CF-JEPA와 같은 GPU에서 동시 실행)
- 바꾼 것: ρ를 0에서 시작해 선형으로 키우는 warmup(`gate_sparsity_warmup_steps`, 진단은 1,600 step, 기본값 6,600), 시작 α = σ(2) ≈ 0.88, ρ = 1e-4.
- 명령: `bash scripts/colab/train_cfjepa.sh model=cfjepa_gate data=collision_tworoom output_model_name=gate_diag trainer.max_epochs=1 model.gate_sparsity_warmup_steps=1600`
- 결과: α가 0.89 → 0.99로 올라 **모든 샘플에서 열렸다**(open ratio 1.0). E5a와 반대로 1에 포화했다.

**α–contact 분석** (`scripts/analyze_gate.py`, data 마지막 500 episode, 15,399 transition, contact transition 16.2%)
- Transition 라벨: frame s+5k → s+5k+5 예측의 contact는 `contact[s+5k+1 .. s+5k+5]` 중 하나라도 True인지다. `World.collect`가 row j에 step j-1 이후의 frame과 그 step의 info를 저장하고 action은 한 칸 당겨 저장하기 때문이다(stable-worldmodel 0.1.1 `world.py`에서 확인).

| 모델 | α 분포 (중앙값) | AUROC(α → contact) | z^c 오차 non-contact | z^c 오차 contact | 비율 |
|---|---|---|---|---|---|
| E5a (닫힘, 10 epoch) | 3.8e-7 | **0.47** | 0.0054 | 0.0356 | **6.6×** |
| E5b (열림, 1 epoch) | 0.988 | **0.54** | 0.0309 | 0.0680 | 2.2× |

- **Gate는 contact를 구분하지 못한다.** 둘 다 무작위(0.5) 수준이다. 0이나 1 근처의 작은 차이 안에서도 contact 순서를 따르지 않는다.
- **E5a(사실상 CF-JEPA)의 z^c 예측 오차는 contact 구간에서 6.6배 크다.** E2에서 ground-truth state로 보인 irreducible error를 학습된 latent에서도 확인한 것이다. 같은 data로 학습한 CF-JEPA(아래 남은 일)로 확정한다.
- E5b의 오차가 전반적으로 큰 것은 1 epoch만 학습했기 때문이다. E5a와 직접 비교하지 않는다.

**Latent probe** (`scripts/probe_latent.py`, E5a checkpoint, 같은 500 episode의 frame, episode 기준 80/20 분할)

| target | z^c (linear / MLP) | z^u (linear / MLP) | z 전체 (linear / MLP) |
|---|---|---|---|
| obstacle x (R²) | 0.756 / 0.846 | 0.564 / 0.927 | 0.817 / 0.937 |
| obstacle y (R²) | 0.422 / 0.649 | 0.260 / 0.861 | 0.618 / 0.872 |
| agent x (R²) | 0.999 / 0.999 | 0.984 / 0.997 | 0.999 / 0.998 |
| agent y (R²) | 1.000 / 0.999 | 0.990 / 0.997 | 1.000 / 0.999 |
| agent–obstacle 거리 (R²) | 0.542 / 0.751 | 0.362 / 0.876 | 0.632 / 0.893 |
| contact (AUROC) | 0.989 / 0.989 | 0.970 / 0.987 | 0.989 / 0.988 |

- **Contact는 latent에서 거의 완벽하게 읽힌다.** z^c만으로도 linear AUROC 0.989다. Obstacle 위치도 z^u에 비선형으로 들어 있다(MLP R² 0.93 / 0.86). Gate 입력에 필요한 정보는 있다.
- **Agent 위치가 z^u에도 거의 완벽하게 들어 있다(R² 0.98~0.99).** Adversarial loss는 z^u에서 action 정보만 지우고 agent 위치는 지우지 않는다. 그래서 z^u는 순수한 "uncontrollable" 정보가 아니다. 이 때문에 h_int가 충돌과 무관하게 z^c 예측을 보조할 수 있고, E5b에서 gate가 항상 열린 것과 맞는다.
- 주의: probe의 contact는 "이 frame을 만든 step의 contact"(현재)이고, gate가 맞혀야 하는 것은 "다음 5 step 안의 contact"(가까운 미래)다. Contact는 보통 몇 step 이어져서 둘은 강하게 연관되지만 같지는 않다.
- MLP probe는 처음에 과적합했다(합성 data에서 음수 R²). 학습 data의 10%로 early stopping을 넣어 고쳤다.

**해석**
- 실패 원인은 입력 정보나 gate의 표현력이 아니라 목적함수다. `예측 loss + ρ·E[α]`로는 gate가 "contact 때만 열어야 할" 학습 신호를 받지 못한다.
  1. Sigmoid 포화: α가 0이나 1 근처에 가면 기울기 α(1−α)가 사라져, 예측 loss도 sparsity도 gate를 움직이지 못한다. 처음 기운 쪽(E5a는 0, E5b는 1)에 붙는다. E5a는 h_int 출력층을 0으로 초기화해서 초반에 gate가 예측 쪽 gradient를 전혀 받지 못한 것도 원인이다.
  2. h_int가 여분 용량으로 쓰인다: h_int도 z^c, action, 그리고 agent 위치가 담긴 z^u를 보므로, contact와 무관하게 예측을 조금씩 돕는다. ρ가 작으면 항상 여는 것이 이득이다.

**남은 일과 다음 방향**
- 같은 data로 학습한 CF-JEPA baseline(`cfjepa_collision_n2`)을 학습 중이다. 끝나면 `analyze_gate.py`, `probe_latent.py`, `eval.py --config-name=collision_tworoom`으로 비교한다.
- Gate 학습 신호를 바꾼다. 후보는 두 가지다.
  - **이득 기반 gate (비지도)**: 샘플마다 h_int를 켰을 때와 껐을 때의 예측 오차를 함께 계산하고, 켰을 때 오차가 충분히 줄어든 샘플에서만 gate가 1을 목표로 학습한다. Contact 라벨을 쓰지 않으므로 RQ2 취지를 유지한다.
  - **Contact 라벨 지도 gate (상한선)**: gate를 contact 라벨로 직접 학습한다. RQ2에는 답하지 못하지만, gate 구조가 Collision env의 예측과 planning을 얼마나 개선할 수 있는지(H1)의 상한을 보여 준다.
- 분석 data(마지막 500 episode)는 학습 data에도 들어 있다(학습/검증 분할이 clip 단위다). 발표용 수치는 별도 평가 data로 다시 계산한다.

**결과물:** checkpoint `gate_collision_n2/weights_epoch_{1..10}.pt`, `gate_diag/weights_epoch_1.pt`와 metrics(`/content/gate_collision_n2_metrics.csv`, `/content/gate_diag_metrics.csv`, `an_gate_main.csv`, `an_gate_diag.csv`)는 강연우의 Colab VM에만 있다. Proton Drive `results/E5_gate/`에 올려야 한다.
