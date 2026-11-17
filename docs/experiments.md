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
- 논문 조건의 LeWM baseline은 E5에서 학습했다(N=2 평가 0.30).
- Planning cost를 LeWM 방식(마지막 step)으로 바꾸면 같은 checkpoint가 N=2에서 1.00이다(E6).
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

## 2026-11-01 · E5. LeWM을 distractor(N=2) 화면으로 학습

**목적:** E3에서 공개 LeWM(깨끗한 data로 학습)은 N=2에서 0.84로 거의 무너지지 않았다. 논문 Table III는 "All models were trained with images augmented with N=2 distractors"라서 조건이 다르다. 논문과 같은 조건의 LeWM baseline을 직접 학습해, CF-JEPA와 공정하게 비교한다.

**설정:** E3와 학습 data, distractor augmentation, epoch, batch, optimizer, seed(3072)가 모두 같고 모델만 LeWM이다(loss = L_fwd + 0.09·SIGReg).
- 명령: `bash scripts/colab/train_cfjepa.sh model=lewm output_model_name=lewm_tworoom_n2`
- Colab A100, 11/1 11:29~16:43 KST(약 5시간 10분).
- Validation loss (epoch 1 → 10): fwd 0.029 → 0.008, SIGReg 7.3 → 1.46. 학습 자체는 정상적으로 수렴했다.

**결과** (CEM 300×30, 50 episodes, eval seed 42, E3와 같은 시작점)

| 모델 | 학습 data | 평가 N=0 | 평가 N=2 |
|---|---|---|---|
| CF-JEPA (E3) | N=2 augment | 0.94 | **0.88** (논문 0.95) |
| LeWM (E5) | N=2 augment | 0.34 | **0.30** (논문 0.33) |
| LeWM (공개 checkpoint, E0) | 깨끗한 data | 0.86 | 0.84 |

- **논문의 LeWM 붕괴(0.33)를 직접 재현했다(0.30).** 같은 조건에서 CF-JEPA는 0.88이라, 논문의 핵심 비교(0.33 대 0.95)가 우리 실험에서도 나온다.
- Distractor로 학습한 LeWM은 사각형이 없는 화면(N=0)에서도 0.34다. 즉 평가 화면의 사각형 때문이 아니라, 학습 중 latent가 사각형 움직임을 담게 되어 planning cost가 망가진 것으로 해석한다. 학습 loss는 정상적으로 낮아졌으므로, 예측을 못 하는 것이 아니라 planning에 쓸모없는 사각형까지 latent에 담는 것이 문제로 보인다.
- 이 해석은 수치에서 나온 추정이다. latent를 직접 분석하지는 않았다.

**결과물**
- Checkpoint: Proton Drive `results/E5_lewm_tworoom_n2/weights_epoch_10.pt`, `config.json`. 다른 epoch는 설규원 로컬 `checkpoints/lewm_tworoom_n2/`에 있다.
- 평가 결과와 영상: Proton Drive `results/E5_lewm_tworoom_n2/eval/`, 로컬 `outputs/lewm_n2/eval/`.

**남은 문제**
- Seed 1개다(E3와 같음).

## 2026-11-01 · E6. CF-JEPA 평가 설정 튜닝 (재학습 없음)

**목적:** E3의 N=2 0.88을 논문 값 0.95에 가깝게 올린다. 재학습(1회 5시간) 전에, 학습 없이 바꿀 수 있는 평가 설정부터 확인한다.

**방법**
- 보고용 시작점(eval seed 42, 50개)으로 튜닝하면 그 50개에 맞춘 결과가 된다. 그래서 튜닝은 seed 1~4(각 50, 합 200)로 하고, 고른 설정만 seed 42로 한 번 평가했다.
- 비교한 것: checkpoint epoch(6, 8, 10), planning cost 두 가지.
  - `sum_l2`: 논문 Eq. 7. 매 예측 step의 ||z^c_k − z^c_goal||를 더한다(E3 기본값).
  - `last_mse`: LeWM 기본 criterion. 마지막 step의 squared error만 쓴다. E5의 LeWM도 이 cost로 평가했다.
- 명령: `python scripts/colab/tune_eval.py cfjepa_tworoom_n2 e10_sum=10:sum_l2 e10_last=10:last_mse e8_sum=8:sum_l2 e6_sum=6:sum_l2`
- 평가 때 cost만 바꾸는 옵션: `scripts/eval.py ... +cost=last_mse`

**튜닝 결과** (N=2, seed 1~4, 200 episodes)

| 설정 | 평균 | seed별 |
|---|---|---|
| epoch 10, sum_l2 (E3) | 0.89 | 0.96 / 0.82 / 0.92 / 0.86 |
| epoch 10, last_mse | **0.99** | 0.98 / 1.00 / 1.00 / 0.98 |
| epoch 8, sum_l2 | 0.88 | 0.92 / 0.82 / 0.92 / 0.86 |
| epoch 6, sum_l2 | 0.86 | 0.90 / 0.78 / 0.92 / 0.84 |

**보고용 시작점 결과** (seed 42, epoch 10, last_mse): N=2 **1.00**, N=0 **1.00**

**해석**
- 차이는 학습이 아니라 planning cost에서 나왔다. 같은 checkpoint가 cost만 바꿔 0.89 → 0.99가 된다. Epoch은 결과에 거의 영향이 없다.
- `sum_l2`는 중간 step도 goal에 가까워야 cost가 낮다. TwoRoom은 벽을 돌아 문으로 가야 해서 중간에 goal에서 멀어지는 경로가 정답일 때가 있는데, 이 cost는 그런 경로를 불리하게 본다고 추정한다(확인하지 않음).
- 논문 Eq. 7은 분명히 합이다. 그래서 "논문 그대로 구현하면 0.88~0.89, LeWM 기본 cost를 쓰면 0.99~1.00"으로 둘 다 기록한다. 논문 저자의 실제 코드가 어느 쪽인지는 알 수 없다(코드 미공개).
- E5의 LeWM(0.30)도 `last_mse`로 평가했으므로, **같은 cost로 비교하면 CF-JEPA 1.00 대 LeWM 0.30**이다.
- 1.00은 50 episode 기준이다. 튜닝 seed에서도 0.98~1.00이라 우연은 아니지만, 성공률 1.0 근처에서는 더 이상 차이를 볼 수 없다(천장 효과).

**결정 제안 (팀 확인 필요):** 이후 gate 실험의 CEM 평가는 `+cost=last_mse`로 통일한다. 코드 기본값은 논문대로 `sum_l2`로 둔다.

**결과물:** 평가 txt와 영상은 Proton Drive `results/E6_cfjepa_tuning/`, 로컬 `outputs/cfjepa/tune/`에 있다.

## 2026-11-01 · E7. PushT: Distractor env에서 CF-JEPA와 LeWM (N=2)

**목적:** 교수님 요청("기존 benchmark 수치를 여러 개")에 맞춰 TwoRoom 외의 기존 task를 하나 더 재현한다. PushT는 CF-JEPA 논문에서 차이가 가장 큰 task다(Table III, N=2: LeWM 0.01, CF-JEPA 0.87).

**설정** (E3, E5와 같고 data만 다르다)
- Data: HF `quentinll/lewm-pusht`의 `pusht_expert_train.h5`(압축 13GB, 풀면 46GB). `TASK=pusht bash scripts/colab/setup_lewm.sh`로 받는다.
- Env: `envs/distractor_pusht.py`(`iawm/PushTDistractor-v1`). TwoRoom과 같은 사각형을 `DistractorMixin`(`envs/distractor_mixin.py`)으로 붙인다.
- 학습: inv_weight 3(논문 PushT 값, TwoRoom과 같음), 10 epochs, batch 128, seed 3072, Colab A100.
  - `bash scripts/colab/train_cfjepa.sh data=pusht model=cfjepa output_model_name=cfjepa_pusht_n2`
  - `bash scripts/colab/train_cfjepa.sh data=pusht model=lewm output_model_name=lewm_pusht_n2`
- 평가: `bash scripts/colab/eval_cfjepa.sh <ckpt> <N> --config-name=pusht` (CEM 300×30, 50 episodes, eval seed 42).

**확인** (`scripts/colab/smoke_pusht.py`, 통과)
- N=0이면 원래 `swm/PushT-v1`과 frame이 같다.
- N=2이면 화면의 약 4.6%가 바뀌고, goal image는 원래 PushT와 같다(깨끗하다).
- 그림: `outputs/pusht_distractor_sheet.png`(연속 20 step).
- 한계: 사각형이 가장자리에 붙어 있어서, agent(파란 점)가 가장자리 근처에 가면 가려질 수 있다. 논문은 "task를 가리지 않게" 배치했다고만 적었다.

**공개 LeWM checkpoint (깨끗한 data로 학습)로 pipeline 확인**
- N=0: **0.94** (논문 0.96), N=2: 0.92. 평가 pipeline이 PushT에서도 정상이다. TwoRoom(E3)과 마찬가지로 깨끗하게 학습한 LeWM은 사각형에 거의 영향을 받지 않는다.

**학습 진행** (11/1 21:00 시작)
- 1 epoch는 13,933 step이다. 처음에는 약 2.1 it/s였다(TwoRoom 3.3 it/s). GPU 사용률이 13~85%로 오르내렸고, 46GB data를 읽는 data loader가 병목이었다. page cache가 차면 3.4~3.7 it/s까지 오른다.
- **Colab 세션이 학습 도중 세 번 사라졌다.**
  - CF-JEPA 세션: 11/1 22:00 전후와 23:30 무렵에 사라졌다. A100 두 대를 동시에 쓰던 때다.
  - LeWM 세션: 혼자 돌던 11/2 04:50 무렵에도 사라졌다(epoch 5 완료 후).
  - 원인은 확인하지 못했다. A100 한 대만 써도 생기므로, 언제든 세션이 끊길 수 있다고 보고 대응한다.
- **대응: 끊긴 지점부터 이어서 학습(resume)한다.**
  - spt는 epoch마다 optimizer 상태까지 담긴 전체 checkpoint(`/root/.cache/stable-pretraining/runs/.../checkpoints/epoch=K-step=N.ckpt`)를 남긴다. 이것을 epoch마다 로컬로 받아 둔다.
  - 새 세션에서는 이 파일을 `$STABLEWM_HOME/checkpoints/<output_model_name>/<output_model_name>_weights.ckpt`에 두고 같은 명령으로 학습을 시작한다. `scripts/train.py`가 이 파일을 찾아 이어서 학습한다. 11/2 10:27에 LeWM을 epoch 5 끝에서 이어받았다(log: "Restored all states from the checkpoint").
  - Colab CLI의 upload에는 크기 제한이 있고(217MB 파일은 500 오류), 없는 폴더로 올려도 같은 오류가 난다. 그래서 45MB 조각으로 `/content`에 올린 뒤 Colab에서 합친다.
  - 관리 script(로컬 `outputs/pusht_manager/`, git 제외)가 15분마다 세션을 확인한다. 세션이 사라지거나 학습이 죽거나 30분 넘게 멈추면, 새 세션을 만들어 이어서 학습한다. 복구는 job당 3회까지만 한다.
- 순서: LeWM을 먼저 끝내고(11/2 16시 예상), 그다음 CF-JEPA를 새 세션에서 학습한다(11/3 새벽 예상). A100은 한 번에 한 대만 쓴다.

- 학습 완료: LeWM 11/2 16:22, CF-JEPA 11/3 05:23(세션 끊김 없음). 평가는 11/3 05:56에 끝났다.

**결과** (success rate, 50 episodes, eval seed 42, 두 모델 모두 N=2로 학습한 epoch 10)

| 모델 | 평가 N=2 | 평가 N=0 | 논문 N=2 (Table III) |
|---|---|---|---|
| LeWM (`lewm_pusht_n2`) | **0.02** | 0.02 | 0.01 |
| CF-JEPA, cost `sum_l2` (논문 Eq. 7) | 0.70 | 0.72 | 0.87 |
| CF-JEPA, cost `last_mse` (`+cost=last_mse`) | **0.78** | 0.80 | — |
| (참고) 공개 LeWM, 깨끗한 data로 학습 | 0.92 | 0.94 | — |

- **논문의 핵심 차이를 재현했다.** 사각형이 있는 data로 학습하면 LeWM은 거의 실패하고(0.02), CF-JEPA는 0.70~0.78을 유지한다. TwoRoom(E3~E6)에 이어 두 번째 기존 benchmark다.
- LeWM은 깨끗한 환경(N=0)에서도 0.02다. 사각형 때문에 planning이 흔들리는 수준이 아니라, 사각형이 있는 data로 학습한 것 자체가 표현을 망가뜨렸다는 뜻이다. Forward loss는 0.006까지 내려갔으므로, 학습이 안 된 것이 아니라 사각형 움직임을 예측하는 데 용량을 썼다고 본다. 깨끗하게 학습한 공개 LeWM이 N=2에서 0.92인 것과 대조된다.
- CF-JEPA는 논문(0.87)보다 낮다. `last_mse` cost가 `sum_l2`보다 0.08 높은 것은 TwoRoom 튜닝(E6)과 같은 방향이다. 차이의 후보는 seed 1개(50 episodes의 표준오차 약 ±0.06), 코드 비공개로 생긴 구현 차이(distractor 배치, data 순서) 등이다.
- 아직 하지 않은 것: seed 반복, eval episode 늘리기.

**결과물:** Proton Drive `results/E7_pusht/`에 두 모델의 `weights_epoch_10.pt`와 config, 평가 결과 txt 6개, 평가 영상 일부가 있다. Colab VM은 평가 직후 종료했다.

## 2026-11-01 · E8. Interaction gate 첫 구현과 학습: collapse, 포화, probe

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

**E8a. 첫 정식 학습** (`gate_collision_n2`, 10 epoch, 3시간 31분)
- 설정: ρ = 1e-3, 시작 α = σ(0) = 0.5, warmup 없음.
- 명령: `bash scripts/colab/train_cfjepa.sh model=cfjepa_gate data=collision_tworoom output_model_name=gate_collision_n2`
- 결과 (validation, epoch 1 → 10): fwd 0.078 → 0.020, inv 0.82 → 0.77, adv ≈ 1.0. **α 평균 0.0006 → 0.0000, 최대값도 0.0000이다. Gate가 첫 epoch 안에 완전히 닫혔다(collapse).** 결과적으로 h_int가 꺼진 CF-JEPA와 같다.
- 300 step 시험 학습에서는 α가 0.5 → 0.47로 천천히 내려가 문제가 없어 보였다. 300 step(전체의 1%)으로 판단한 것이 잘못이었다.

**E8b. 진단 학습** (`gate_diag`, 1 epoch, 약 40분, 비교용 CF-JEPA와 같은 GPU에서 동시 실행)
- 바꾼 것: ρ를 0에서 시작해 선형으로 키우는 warmup(`gate_sparsity_warmup_steps`, 진단은 1,600 step, 기본값 6,600), 시작 α = σ(2) ≈ 0.88, ρ = 1e-4.
- 명령: `bash scripts/colab/train_cfjepa.sh model=cfjepa_gate data=collision_tworoom output_model_name=gate_diag trainer.max_epochs=1 model.gate_sparsity_warmup_steps=1600`
- 결과: α가 0.89 → 0.99로 올라 **모든 샘플에서 열렸다**(open ratio 1.0). E8a와 반대로 1에 포화했다.

**α–contact 분석** (`scripts/analyze_gate.py`, data 마지막 500 episode, 15,399 transition, contact transition 16.2%)
- Transition 라벨: frame s+5k → s+5k+5 예측의 contact는 `contact[s+5k+1 .. s+5k+5]` 중 하나라도 True인지다. `World.collect`가 row j에 step j-1 이후의 frame과 그 step의 info를 저장하고 action은 한 칸 당겨 저장하기 때문이다(stable-worldmodel 0.1.1 `world.py`에서 확인).

| 모델 | α 분포 (중앙값) | AUROC(α → contact) | z^c 오차 non-contact | z^c 오차 contact | 비율 |
|---|---|---|---|---|---|
| E8a (닫힘, 10 epoch) | 3.8e-7 | **0.47** | 0.0054 | 0.0356 | **6.6×** |
| E8b (열림, 1 epoch) | 0.988 | **0.54** | 0.0309 | 0.0680 | 2.2× |

- **Gate는 contact를 구분하지 못한다.** 둘 다 무작위(0.5) 수준이다. 0이나 1 근처의 작은 차이 안에서도 contact 순서를 따르지 않는다.
- **E8a(사실상 CF-JEPA)의 z^c 예측 오차는 contact 구간에서 6.6배 크다.** E2에서 ground-truth state로 보인 irreducible error를 학습된 latent에서도 확인한 것이다. 같은 data로 학습한 CF-JEPA(아래 남은 일)로 확정한다.
- E8b의 오차가 전반적으로 큰 것은 1 epoch만 학습했기 때문이다. E8a와 직접 비교하지 않는다.

**Latent probe** (`scripts/probe_latent.py`, E8a checkpoint, 같은 500 episode의 frame, episode 기준 80/20 분할)

| target | z^c (linear / MLP) | z^u (linear / MLP) | z 전체 (linear / MLP) |
|---|---|---|---|
| obstacle x (R²) | 0.756 / 0.846 | 0.564 / 0.927 | 0.817 / 0.937 |
| obstacle y (R²) | 0.422 / 0.649 | 0.260 / 0.861 | 0.618 / 0.872 |
| agent x (R²) | 0.999 / 0.999 | 0.984 / 0.997 | 0.999 / 0.998 |
| agent y (R²) | 1.000 / 0.999 | 0.990 / 0.997 | 1.000 / 0.999 |
| agent–obstacle 거리 (R²) | 0.542 / 0.751 | 0.362 / 0.876 | 0.632 / 0.893 |
| contact (AUROC) | 0.989 / 0.989 | 0.970 / 0.987 | 0.989 / 0.988 |

- **Contact는 latent에서 거의 완벽하게 읽힌다.** z^c만으로도 linear AUROC 0.989다. Obstacle 위치도 z^u에 비선형으로 들어 있다(MLP R² 0.93 / 0.86). Gate 입력에 필요한 정보는 있다.
- **Agent 위치가 z^u에도 거의 완벽하게 들어 있다(R² 0.98~0.99).** Adversarial loss는 z^u에서 action 정보만 지우고 agent 위치는 지우지 않는다. 그래서 z^u는 순수한 "uncontrollable" 정보가 아니다. 이 때문에 h_int가 충돌과 무관하게 z^c 예측을 보조할 수 있고, E8b에서 gate가 항상 열린 것과 맞는다.
- 주의: probe의 contact는 "이 frame을 만든 step의 contact"(현재)이고, gate가 맞혀야 하는 것은 "다음 5 step 안의 contact"(가까운 미래)다. Contact는 보통 몇 step 이어져서 둘은 강하게 연관되지만 같지는 않다.
- MLP probe는 처음에 과적합했다(합성 data에서 음수 R²). 학습 data의 10%로 early stopping을 넣어 고쳤다.

**해석**
- 실패 원인은 입력 정보나 gate의 표현력이 아니라 목적함수다. `예측 loss + ρ·E[α]`로는 gate가 "contact 때만 열어야 할" 학습 신호를 받지 못한다.
  1. Sigmoid 포화: α가 0이나 1 근처에 가면 기울기 α(1−α)가 사라져, 예측 loss도 sparsity도 gate를 움직이지 못한다. 처음 기운 쪽(E8a는 0, E8b는 1)에 붙는다. E8a는 h_int 출력층을 0으로 초기화해서 초반에 gate가 예측 쪽 gradient를 전혀 받지 못한 것도 원인이다.
  2. h_int가 여분 용량으로 쓰인다: h_int도 z^c, action, 그리고 agent 위치가 담긴 z^u를 보므로, contact와 무관하게 예측을 조금씩 돕는다. ρ가 작으면 항상 여는 것이 이득이다.

**남은 일과 다음 방향**
- 같은 data로 학습한 CF-JEPA baseline(`cfjepa_collision_n2`)을 학습 중이다. 끝나면 `analyze_gate.py`, `probe_latent.py`, `eval.py --config-name=collision_tworoom`으로 비교한다.
- Gate 학습 신호를 바꾼다. 후보는 두 가지다.
  - **이득 기반 gate (비지도)**: 샘플마다 h_int를 켰을 때와 껐을 때의 예측 오차를 함께 계산하고, 켰을 때 오차가 충분히 줄어든 샘플에서만 gate가 1을 목표로 학습한다. Contact 라벨을 쓰지 않으므로 RQ2 취지를 유지한다.
  - **Contact 라벨 지도 gate (상한선)**: gate를 contact 라벨로 직접 학습한다. RQ2에는 답하지 못하지만, gate 구조가 Collision env의 예측과 planning을 얼마나 개선할 수 있는지(H1)의 상한을 보여 준다.
- 분석 data(마지막 500 episode)는 학습 data에도 들어 있다(학습/검증 분할이 clip 단위다). 발표용 수치는 별도 평가 data로 다시 계산한다.

**결과물:** Proton Drive `results/E8_gate/`에 `weights_epoch_10.pt`, `config.json`(E8a), `gate_collision_n2_metrics.csv`, `gate_diag_metrics.csv`가 있다. `gate_diag` checkpoint와 E8a의 epoch 1~9 checkpoint는 받지 않았다.

## 2026-11-01~02 · E9. Gate 학습 신호 바꾸기(이득 기반, contact 지도)와 잔차 probe

**목적:** E8에서 sparsity 방식의 gate가 0이나 1에 포화하고 contact를 구분하지 못했다(AUROC 0.47 / 0.54). Gate의 학습 신호를 바꿔 contact 때만 열리는지, 열렸을 때 예측이 좋아지는지 확인한다. 같은 data로 학습한 CF-JEPA baseline도 만든다.

**구현** (`src/iawm/cfjepa_gate.py`, `scripts/train.py`의 `[interaction gate]` 블록)
- `predictor.gate_objective`: `sparsity`(E8), `gain`, `supervised`.
- `gain`과 `supervised`에서는 gate를 BCE로만 학습한다. Gate 입력과, 예측에 쓰는 α를 detach한다. 그래서 예측 loss가 gate를 움직이지 않고, gate BCE가 encoder를 움직이지 않는다.
- h_int는 α와 무관한 보조 loss(`int_aux_loss`, α=1 예측의 z^c 오차)로도 학습한다. Gate가 닫혀도 h_int가 계속 배운다.
- Gate 목표값:
  - `gain`: h_int를 켰을 때 z^c 오차가 τ(`gate_gain_threshold`, 기본 0.3) 이상 줄었으면 1. Contact 라벨을 쓰지 않는다.
  - `supervised`: transition(frame t → t+5) 동안 contact가 있었으면 1.
- `supervised` 라벨: data loader는 action 외 column을 frameskip 간격으로 건너뛰어 읽는다. 그래서 `scripts/add_contact_labels.py`가 `contact_next[j] = any(contact[j+1 .. j+5])`를 미리 계산해 h5에 더한다(`analyze_gate.py`의 transition 라벨과 같은 정의, 가짜 data로 일치 확인). 원본은 학습 중인 다른 process가 읽고 있어서, 사본 `collision_tworoom_contact.h5`에 쓰고 `configs/train/data/collision_tworoom_contact.yaml`이 읽는다. train.py가 이 column도 z-score 정규화하므로 gate 블록은 0보다 큰 값을 1로 본다.
- 새 로깅: `gate_target_ratio`, `gate_acc`, `int_rel_gain`, `int_aux_loss`, `gate_bce_loss`.
- 검증: `smoke_gate.py`에 두 방식의 gradient 격리(fwd loss → gate 없음, gate BCE → encoder 없음, h_int 보조 loss → g 없음)를 추가했고 모두 통과했다. 가짜 h5로 `train.py`를 끝까지 돌려 `contact_next`가 정규화를 거쳐도 라벨로 해석되는 것을 확인했다.
- `scripts/analyze_gate.py`: h_int를 끈 예측(`err_c_off`)과 켠 예측(`err_c_on`, α=1)의 오차, contact별 상대 이득, τ별 목표 비율을 출력한다.
- `scripts/probe_residual.py`(새 파일): 아래 잔차 probe.

**학습** (Colab A100 40GB 1장, seed 3072, 10 epoch, batch 128, Distractor N=2 augment, `data=collision_tworoom`)

| 이름 | 모델 | 비고 |
|---|---|---|
| `cfjepa_collision_n2` | CF-JEPA (`model=cfjepa`, 기본값) | Gate 비교용 baseline. 완료 |
| `gate_gain_n2` | `gate_objective=gain`, τ=0.3 | Epoch 6.5에서 중단(아래). `weights_epoch_6.pt` 사용 |
| `gate_sup_n2` | `gate_objective=supervised`, `data=collision_tworoom_contact` | 완료 |

- 명령: `bash scripts/colab/train_cfjepa.sh model=cfjepa_gate data=collision_tworoom model.predictor.gate_objective=gain output_model_name=gate_gain_n2` (supervised는 `data=collision_tworoom_contact`, `gate_objective=supervised`)
- 세 학습이 한 GPU를 나눠 쓴 구간이 있어 epoch당 시간이 20분(단독)에서 35~40분으로 늘었다.

**이득 기반 gate: 목표값이 한 번도 1이 되지 않았다**
- Validation `gate_target_ratio`가 epoch 1~6 내내 0.0이었다. h_int가 z^c 오차를 30% 이상 줄인 샘플이 없었다는 뜻이다. Gate는 항상 닫혔고(`gate_open_ratio` 0), `gate_acc` 1.0은 전부 0인 목표를 맞힌 것이라 의미가 없다. 그래서 epoch 6.5에서 중단했다.
- `analyze_gate.py` (`gate_gain_n2/weights_epoch_6.pt`, 마지막 500 episode):

| | err_off (h_int 끔) | err_on (α=1) | 상대 이득 평균 |
|---|---|---|---|
| non-contact | 0.00984 | 0.00977 | 0.94% |
| contact | 0.04376 | 0.04357 | 0.44% |

  - h_int는 contact 구간 오차를 0.44%만 줄였다. 오히려 non-contact에서 더 도왔다. 상대 이득으로 contact를 구분하는 AUROC는 0.348, 절대 이득은 0.605다.
  - τ를 낮춰도 구분되지 않는다. τ=0.02에서 목표=1 비율은 non-contact 13.1%, contact 6.0%이고, τ≥0.05에서는 거의 0이다.

**Contact 지도 gate: 스위치는 되지만 예측은 그대로다** (`gate_sup_n2/weights_epoch_10.pt`, 마지막 500 episode)

| | non-contact | contact |
|---|---|---|
| α 평균 | 0.094 | 0.514 |
| 열린 비율 (α > 0.5) | 2.2% | 50.8% |
| 실제 z^c 예측 오차 | 0.00545 | 0.03477 |
| h_int 끔 / 켬 (α=1) | 0.00545 / 0.00540 | 0.03493 / 0.03464 |

- **AUROC(α → contact) = 0.899.** Contact 때 열리고 평소에는 닫힌다. 다만 라벨로 학습했으므로 RQ2의 답이 아니라 "스위치 구조가 가능하다"는 상한선이다.
- **예측은 좋아지지 않았다.** Contact 구간 실제 오차 0.0348은 h_int를 끈 예측(0.0349)과 같다. α=1로 켜도 0.8%만 준다. Gate가 제대로 열려도, 열렸을 때 고칠 h_int가 contact 효과를 배우지 못했다.
- Contact 구간 오차는 non-contact의 6.4배다(E8a의 6.6배와 같은 양상).

**잔차 probe: 병목은 latent의 obstacle 정보다** (`scripts/probe_residual.py`, `cfjepa_collision_n2/weights_epoch_10.pt` 고정)
- 방법: clip의 마지막 position(history 3 frame을 모두 본 예측)에서 잔차 r = z^c_{t+1} − (CF-JEPA z^c 예측)을 구하고, 큰 MLP(hidden 512, early stopping)로 r을 맞힌다. Episode 기준 80/20 분할의 test에서 오차 감소율을 본다.
  - 입력 `zc`: z^c_{t-2..t} + action / `zc+zu`: z^c, z^u_{t-2..t} + action (h_int가 볼 수 있는 정보) / `oracle`: agent 위치_{t-2..t}, obstacle 위치·속도_t + action (ground-truth state).
  - 지표 B = 1 − |r − r̂|² / |r − group 평균 r|². 잔차의 평균(bias)만 맞혀도 |r|²이 크게 줄어서(랜덤 모델에서 0.97), group별 평균을 빼고 남는 오차 대비로 본다. 주 지표다.
- 결과 (B, 괄호 안은 |r|² 대비 A):

| 입력 | 1,500 episode: non-contact | 1,500: contact | **10,000 episode: non-contact** | **10,000: contact** |
|---|---|---|---|---|
| zc | 0.247 (0.354) | −0.075 (0.015) | 0.369 (0.457) | 0.031 (0.105) |
| zc+zu | 0.230 (0.339) | −0.053 (0.034) | 0.311 (0.406) | 0.046 (0.120) |
| oracle | 0.206 (0.319) | 0.016 (0.098) | 0.417 (0.497) | **0.405** (0.451) |

- 1,500 episode에서는 oracle도 contact 잔차를 맞히지 못했다(contact 학습 샘플 약 1,700개). 10,000 episode로 늘리니 oracle은 contact 잔차의 40%를 설명한다. 즉 contact 효과는 정확한 state가 있으면 예측할 수 있다.
- **그런데 latent(zc+zu)로는 4.6%만 설명한다.** 충분한 data와 큰 MLP로도 이 정도이므로, h_int를 키워도 크게 나아지지 않는다. E8 probe에서 obstacle 위치가 R² 0.86~0.93(agent는 0.999)으로만 읽힌 것과 맞는다. "다음 5 step에 부딪혀 얼마나 밀릴지"를 계산하기에는 latent의 obstacle 위치가 너무 흐릿하다.
- Non-contact 잔차는 입력과 관계없이 25~40% 설명된다. g가 contact와 무관하게 underfit된 부분이 있고, h_int가 contact보다 non-contact를 더 도운 이유가 이것이다.

**해석**
- 병목은 gate(스위치)도 h_int의 크기도 아니라, encoder가 obstacle 위치·속도를 충분히 정밀하게 담지 않는 것이다.
- 이득 기반 gate가 라벨 없이 열리려면 "켰을 때 contact에서만 크게 이득"이어야 하는데, 지금 latent에서는 그 이득 자체가 없다. 그래서 E8의 sparsity 방식, E9의 gain 방식이 모두 실패했다.

**남은 일**
- Latent에 obstacle 정보를 정밀하게 담는 방법을 찾는다. 후보: z^u 차원 64 → 128, distractor 없이(N=0) 학습해 z^u 용량 경쟁 제거, h_int 보조 loss 비중 높이기, 여러 step 예측 loss. 2~3 epoch 학습 후 `probe_latent.py`, `probe_residual.py`의 `zc+zu` contact B가 4.6%에서 oracle(40%) 쪽으로 오르는지로 비교한다.
- 그 뒤 h_int의 contact 이득이 커지면, 이득 기반 gate를 절대 이득 기준으로 다시 학습해 라벨 없이 AUROC 0.8 이상(H3)이 나오는지 본다. 이번 결과에서도 절대 이득의 AUROC(0.60)가 상대 이득(0.32~0.35)보다 나았다.
- 아직 하지 않은 것: `cfjepa_collision_n2`의 `analyze_gate.py` 결과와 epoch별 loss, 모든 모델의 planning 성공률(`eval.py --config-name=collision_tworoom`, H1), Distractor env 평가(H2), seed 3개, 학습에 쓰지 않은 평가 data.

**결과물:** Proton Drive `results/E9_gate_objectives/`에 checkpoint 3개(`cfjepa_collision_n2_weights_epoch_10.pt`, `gate_sup_n2_weights_epoch_10.pt`, `gate_gain_n2_weights_epoch_6.pt`)와 각 config, 분석 CSV(`an_gate_main.csv`, `an_gate_diag.csv`, `an_gate_gain_e6.csv`, `an_gate_sup.csv`)가 있다. Colab VM은 11/2 오전에 종료했다.

## 2026-11-02 · E10. Encoder가 obstacle 정보를 담게 하는 방법 진단 (3 epoch)

**목적:** E9 잔차 probe에서 병목이 encoder로 나왔다. 충돌 순간 CF-JEPA z^c 잔차를 latent(zc+zu)는 4.6%, oracle state는 40% 설명했다. 이 결과는 2606.30068의 주장과 맞는다. Reward 없는 예측 objective는 control에 필요한 외부(exogenous) feature를 버리고, 라벨 약 2%로 이를 되살릴 수 있다는 내용이다. 짧은 학습(3 epoch)으로 다음 두 후보가 encoder의 obstacle 정보를 늘리는지 비교한다. Gate 재학습(2단계)에 쓸 encoder를 고르는 단계다.
- (A) contact 보조 loss
- (B) distractor 제거

**구현**
- `scripts/train.py` `[contact aux]` 블록: `model.contact_head=true`이면 two-layer MLP 하나가 [z_{t-1}, z_t](z^c+z^u 전체)로부터 `contact_next[t]`(다음 5 step 안 contact)를 BCE로 예측한다. 이 loss는 encoder까지 backprop된다. 가중치는 `model.contact_aux_weight`이고, 라벨은 z-score 정규화되어 있어 0보다 크면 1로 본다. NaN인 row는 loss에서 뺀다.
- `scripts/add_contact_labels.py --label-frac 0.02`: episode 단위로 2%(seed 0, 10,000 중 190 episode, row 1.91%)에만 라벨을 남긴 `contact_next_p2`를 만든다. 나머지 row는 NaN이다.
- `configs/train/data/collision_tworoom_aux.yaml`: `contact_next`, `contact_next_p2`를 함께 읽는다.
- `src/iawm/cfjepa.py`, `cfjepa_gate.py`: `contact_head` 인자를 더했다.

**학습** (Colab A100, `model=cfjepa`, `data=collision_tworoom_aux`, 3 epoch, 나머지는 E9와 같다. 두 학습씩 한 GPU를 나눠 썼다)

| 이름 | 설정 |
|---|---|
| `e10_base` | CF-JEPA 그대로 (3 epoch 기준선) |
| `e10_lab2` | (A) contact 보조 loss, 라벨 2% (`+model.contact_head=true +model.contact_aux_weight=1.0 +model.contact_aux_key=contact_next_p2`) |
| `e10_lab100` | (A') contact 보조 loss, 라벨 100% (`contact_aux_key=contact_next`) |
| `e10_n0` | (B) distractor 없음 (`distractor.num=0`) |

- 명령과 순서: `outputs/e10/e10_pipeline.sh`(gitignored)가 data 수집, 라벨 추가, smoke test, 학습, probe를 차례로 한다. 학습 한 개가 GPU 메모리를 약 13GB 써서, A100 40GB에 4개를 동시에 띄우면 2개가 OOM으로 바로 죽는다. 그래서 2개씩 나눠 돌렸다.
- `e10_base`, `e10_lab2`는 처음 돌릴 때 학습 도중 세션이 꺼졌다. PushT manager(E7)가 LeWM을 마친 뒤 Colab 세션을 전부 종료했기 때문이다. 새 세션에서 data를 다시 만들어 재학습했다. 수집 seed가 같아 data 통계(row 659,451, contact_next 0.1488, p2 190 episode)가 같다.
- 판정: `python scripts/probe_residual.py --ckpt <name>/weights_epoch_3.pt --episodes 10000` (E9와 같은 방법, 지표 B)

**결과** (contact 구간 B. 괄호는 non-contact B. "oracle 대비"는 latent B / oracle B)

| 모델 | zc | zc+zu | oracle | oracle 대비 | |r|² contact / non-contact |
|---|---|---|---|---|---|
| E9 `cfjepa_collision_n2` (10 epoch, 참고) | 0.031 | 0.046 (0.311) | 0.405 | 11% | — |
| `e10_base` | 0.154 | 0.152 (0.533) | 0.504 | 30% | 0.0405 / 0.0083 |
| `e10_lab2` | 0.227 | **0.252** (0.617) | 0.572 | **44%** | 0.0541 / 0.0133 |
| `e10_lab100` | 0.210 | 0.209 (0.523) | 0.593 | 35% | 0.0554 / 0.0122 |
| `e10_n0` | 0.410 | **0.445** (0.823) | 0.449 | **99%** | 0.3719 / 0.2319 |

**해석**
- **학습을 오래 하면 obstacle 정보가 줄어든다.** 같은 CF-JEPA인데 3 epoch(15.2%)가 10 epoch(4.6%)보다 latent의 contact B가 높다. 예측 objective만으로 학습을 계속하면 encoder가 obstacle 정보를 버린다는 2606.30068의 설명과 맞는다. 다만 E9와 E10은 data를 따로 수집했다.
- **(A) contact 라벨 2%는 효과가 있다.** zc+zu B가 0.152에서 0.252로, oracle 대비 30%에서 44%로 올랐다. 라벨 100%(0.209)는 2%보다 오히려 낮다. 라벨 양보다 보조 신호가 있는지 없는지가 중요해 보이며, seed 1개라 2%와 100%의 차이는 noise일 수 있다.
- **그러나 늘어난 정보가 대부분 z^c에 들어갔다.** 라벨 100%에서 zc만 넣은 B(0.210)와 zc+zu(0.209)가 같다. 라벨 2%에서도 z^u가 더한 몫은 0.025뿐이다. 보조 head가 z^c+z^u 전체를 보기 때문에, 정보가 controllable 쪽에 섞여 들어간다. CF-JEPA의 분리 의도와 어긋나는 결과다.
- **(B) distractor를 없애면 latent가 oracle만큼 맞힌다**(oracle 대비 99%). Distractor 사각형과 obstacle이 z^u 용량을 두고 경쟁해서 obstacle 정보가 흐려졌다는 가설과 맞는다. 다만 이 모델은 잔차 크기가 다른 모델의 약 10배여서(non-contact |r|² 0.23 vs 0.01), z^c 예측 자체가 3 epoch에서 덜 학습된 상태다. B는 정규화된 비율이라 비교는 되지만, 같은 조건이라고 보기는 어렵다. 또 distractor가 없으면 Distractor env robustness(H2)를 확인할 수 없으므로, 이 설정을 그대로 최종 모델로 쓸 수는 없다.

**남은 일**
- 2단계 후보:
  - 라벨 2% 보조 loss를 쓰되, head 입력을 z^u(와 z^c의 stop-gradient)로 바꿔 obstacle 정보가 z^u에 들어가게 한다.
  - Distractor 개수를 학습 중에 0에서 2로 늘리는 curriculum.
  - 선택한 encoder로 10 epoch 학습한 뒤 h_int의 contact 이득과 contact 지도·이득 기반 gate를 다시 본다(E9의 0.4~0.8% 대비).
- 아직 하지 않은 것: seed 반복, 10 epoch에서도 효과가 유지되는지, planning 성공률.

**결과물:** Proton Drive `results/E10_encoder_diag/`에 4개 모델의 `weights_epoch_3.pt`와 config, probe 출력(`probe_e10_*.txt`)이 있다. Colab VM은 종료했다.

## 2026-11-03 · E11. Encoder 수정 시도: z^u 전용 보조 loss와 distractor curriculum (3 epoch)

**목적:** E10에서 contact 라벨 2% 보조 loss는 encoder의 obstacle 정보를 늘렸다(oracle 대비 30% → 44%). 그런데 늘어난 정보가 주로 z^c에 들어갔다. 또 distractor를 없애면 latent가 oracle만큼 맞혔지만, 그 설정은 Distractor env와 맞지 않는다. 두 문제를 각각 고친 설정을 E10과 같은 조건으로 비교한다.

**구현** (`scripts/train.py`)
- `+model.contact_aux_input=zu`: contact head 입력에서 z^c를 stop-gradient한다. 그러면 보조 loss의 gradient가 z^u 쪽으로만 encoder에 들어간다. 기본값 `full`은 E10과 같다.
- `+distractor.curriculum_epochs=K`: 학습 epoch e에서 사각형 수를 round(num·e/K)로 정한다(e ≥ K부터 num). Validation은 항상 num이다. K=2, 3 epoch이면 사각형 수가 epoch 0, 1, 2에서 0, 1, 2개다.

**학습** (`model=cfjepa`, `data=collision_tworoom_aux`, 3 epoch, 두 학습이 A100 한 장을 나눠 썼다. 파이프라인은 `outputs/e11/e11_pipeline.sh`)

| 이름 | 설정 |
|---|---|
| `e11_zu` | 라벨 2% 보조 loss, `contact_aux_input=zu` |
| `e11_cur` | 보조 loss 없음, `distractor.curriculum_epochs=2` |

**결과** (contact 구간 B. 괄호는 non-contact B. E10 표와 같은 방법이다)

| 모델 | zc | zc+zu | oracle | oracle 대비 | \|r\|² contact / non-contact |
|---|---|---|---|---|---|
| `e10_base` (참고) | 0.154 | 0.152 (0.533) | 0.504 | 30% | 0.0405 / 0.0083 |
| `e10_lab2` (참고) | 0.227 | 0.252 (0.617) | 0.572 | 44% | 0.0541 / 0.0133 |
| `e11_zu` | 0.151 | 0.133 (0.560) | 0.432 | 31% | 0.0671 / 0.0106 |
| `e11_cur` | 0.007 | 0.033 (0.137) | 0.329 | 10% | 0.1499 / 0.0232 |

**해석**
- **z^u 전용 보조 loss는 효과가 없었다.** 결과가 base와 같다(31% vs 30%). 보조 loss를 z^u로만 흘리면 obstacle 정보가 z^u에 들어가지 않는다. Encoder가 z^u에 obstacle을 담는 데에는 보조 신호만으로 부족한 다른 제약이 있는 것으로 보인다. 예를 들어 adversarial loss가 z^u에서 action 정보를 지우는 과정이 agent 근처 정보를 함께 지우면, contact 예측이 어려워진다.
- **Curriculum은 오히려 나빴다.** 마지막 epoch에 사각형이 갑자기 2개로 늘면서 z^c 예측 자체가 무너졌다. 잔차가 base의 2~4배이고, oracle B도 0.329로 낮다. 3 epoch는 curriculum에 너무 짧다.
- E10의 차이(30%, 35%, 44%)는 seed 1개 결과라 noise일 수 있다. 확실히 큰 효과는 distractor 제거(99%)뿐이다.

**결정:** 2단계(E12)에는 E10 `lab2` 설정(full 입력, 라벨 2%)을 쓴다. 이 설정이 측정된 것 중 가장 좋았다. 또 h_int는 z^c와 z^u를 모두 입력으로 받으므로, obstacle 정보가 z^c에 들어가도 h_int가 쓸 수 있다. z^u 차원을 키우는 시도는 시간과 compute unit 때문에 뒤로 미룬다.

**결과물:** Proton Drive `results/E11_encoder_fix/`에 두 모델의 `weights_epoch_3.pt`와 config, probe 출력이 있다.

## 2026-11-03 · E12. Contact 보조 loss(라벨 2%)를 붙인 gate 모델 학습과 분석

**목적:** E10 `lab2` encoder 수정(contact 보조 loss, full 입력, 라벨 2%)을 gate 모델에 붙여 다시 학습한다. 확인할 것은 두 가지다. 첫째, h_int의 contact 이득이 E9(0.4~0.8%)보다 커지는가. 둘째, 그 이득으로 이득 기반 gate가 라벨 없이 열리는가.

**학습** (Colab A100 한 장에 두 학습, `outputs/e12/e12_pipeline.sh`)
- 공통: `model=cfjepa_gate data=collision_tworoom_aux trainer.max_epochs=10 +model.contact_head=true +model.contact_aux_weight=1.0 +model.contact_aux_key=contact_next_p2`
- `e12_sup`: `model.predictor.gate_objective=supervised`(gate 목표는 contact 라벨 100%. 보조 loss는 라벨 2%)
- `e12_gain`: `model.predictor.gate_objective=gain model.gate_gain_threshold=0.1`(E9의 τ=0.3에서 낮췄다)
- epoch당 약 47분(두 학습이 GPU를 나눠 씀).

**세션 손실과 대응**
- 학습 세션이 epoch 8 직후(11/3 08:30 무렵) 사라졌다. 가중치(`weights_epoch_8.pt`)만 로컬에 받아 두었고, 전체 checkpoint(optimizer 상태)는 받지 않았다. 그래서 이어서 학습할 수 없었고, 분석과 평가는 **epoch 8**로 했다.
- 분석·평가용 세션도 두 번(11:20, 14:15 무렵) 사라졌다. 둘 다 Collision data(1만 episode, 약 1시간)를 수집하던 중이었다. 원인은 확인하지 못했다. "kernel이 쉬면 회수된다"는 가설은 맞지 않았다. 두 번째 세션은 로컬 감시 프로세스가 10분마다 kernel에 명령을 보내는 중에 사라졌다.
- 세 번째 시도에서는 분석·평가용 data를 2,000 episode로 줄였다(`collect_collision.py --episodes 2000`, 약 11분). 분석과 평가에는 시작 위치와 정규화 통계만 필요하다. 수집 seed가 같아서, 학습 data의 일부와 겹친다. 잔차 probe(1만 episode 필요)는 하지 않았다. VM 메모리는 83GB 중 최대 6GB만 썼다.

**분석** (`analyze_gate.py --distractors 2`, 2,000 episode data의 마지막 500 episode)

| | `e12_sup` | `e12_gain` | E9 `gate_sup_n2` (참고) |
|---|---|---|---|
| α 평균 non-contact / contact | 0.108 / 0.518 | 1.1e-5 / 1.2e-5 | 0.094 / 0.514 |
| AUROC(α → contact) | **0.905** | 0.533 | 0.899 |
| h_int 상대 이득 non-contact / contact | 0.39% / **0.49%** | 0.41% / 0.23% | — / 0.8% |
| AUROC(절대 이득 → contact) | 0.619 | 0.542 | — |
| z^c 오차 contact / non-contact | 3.3배 | 2.7배 | 6.4배 |

- **Contact 지도 gate는 E9와 같다.** Contact 때 열리지만(AUROC 0.905), 열어도 z^c 오차가 0.49%만 준다. Encoder 수정이 h_int의 contact 이득을 늘리지 못했다.
- **이득 기반 gate는 계속 닫혀 있다.** τ=0.1에서도 목표가 1인 샘플이 거의 없어서(0.03%) gate가 0으로 collapse했다. 사후 분석에서는 τ를 0.02까지 낮춰도 목표=1 비율이 non-contact 10.9%, contact 5.1%로, 오히려 contact에서 더 낮다.
- Contact/non-contact 오차 비가 E9보다 작다(3.3배 vs 6.4배). 다만 E12 분석은 distractor 2개를 덧그린 이미지에서 했기 때문에(E9는 distractor 없음) non-contact 오차가 커진 영향이 섞여 있다.

**해석:** E10의 잔차 probe 개선(oracle 대비 30% → 44%)은 h_int의 실제 이득으로 이어지지 않았다. 분석 data와 학습 길이가 E10과 다르므로 직접 비교는 어렵다. 하지만 3 epoch에서 본 encoder 개선은 10 epoch 학습에서 유지되지 않았거나, h_int의 용량·학습 신호로는 쓸 수 없는 형태로 보인다.

**결과물:** Proton Drive `results/E12_gate_aux/`에 두 모델의 `weights_epoch_8.pt`와 config, 분석 txt·csv가 있다.

## 2026-11-03 · E13. Gate 모델의 planning 성공률 (H1 Collision env, H2 Distractor env)

**목적:** gate가 실제 control 성공률을 올리는지 본다(H1). 또 Distractor env에서 CF-JEPA의 robustness를 유지하는지 본다(H2). 같은 모델에서 gate만 켜고 끄는 비교(`+gate_mode=off`, h_int를 끄고 g만 씀)를 하고, gate가 없는 원래 CF-JEPA(E9 `cfjepa_collision_n2`, epoch 10)도 함께 평가한다.

**설정**
- H1: `python scripts/eval.py --config-name=collision_tworoom policy=<ckpt> [+gate_mode=off] seed=<42|43|44>`. CEM 300×30, 50 episodes × seed 3개 = 150 episodes. 시작 상태와 goal은 E12의 2,000 episode data에서 뽑는다.
- H2: `python scripts/eval.py policy=<ckpt> [+gate_mode=off] world.num_distractors=2`. Distractor TwoRoom, 50 episodes, seed 42. 모델은 모두 Collision data로 학습했다.
- `scripts/eval.py`에 `+gate_mode` override를 더했다(learned | always | off). 3 episode로 먼저 확인했다.

**결과** (success rate %)

| 모델 | H1 s42 | s43 | s44 | **H1 평균** | **H2 N=2** |
|---|---|---|---|---|---|
| CF-JEPA (E9 `cfjepa_collision_n2`, gate 없음) | 96 | 88 | 90 | **91.3** | 84 |
| `e12_sup`, gate learned | 94 | 90 | 90 | **91.3** | 94 |
| `e12_sup`, gate off | 82 | 82 | 90 | 84.7 | 94 |
| `e12_gain`, gate learned | 88 | 82 | 86 | 85.3 | 90 |
| `e12_gain`, gate off | 84 | 86 | 92 | 87.3 | 92 |

**해석**
- **H1은 확인하지 못했다.** Gate 모델(`e12_sup`, 91.3%)은 gate 없는 CF-JEPA(91.3%)와 같다.
  - `e12_sup`에서 gate를 끄면 84.7%로 떨어진다. 다만 h_int의 예측 이득이 0.5%뿐이라, 이 차이는 h_int가 contact 효과를 맞혀서 생긴 것이라고 보기 어렵다. 학습 중 g와 h_int가 함께 맞춰졌기 때문에, 하나를 빼면 예측이 흔들리는 효과일 가능성이 크다.
  - 50 episodes의 표준오차는 약 ±4%p이고, seed 사이 차이도 최대 8%p다. 그래서 4~6%p 차이는 noise와 구분하기 어렵다.
  - `e12_gain`은 α가 거의 0이라 켜도 꺼도 같아야 한다. 실제 차이(85.3% vs 87.3%)가 noise 수준을 보여 준다.
- **현재 Collision env는 H1을 보이기에 너무 쉽다.** Obstacle을 모르는 CF-JEPA도 91%를 성공한다. 수집 통계(`collect.log`)로는 episode의 73%에서 contact가 일어나지만, 밀린 뒤에도 다시 계획하면 대부분 goal에 도달한다. 차이를 보려면 obstacle이 성공을 실제로 막는 환경이 필요하다. 예를 들어 obstacle 수를 늘리거나, push를 세게 하거나, 시간 제한을 짧게 할 수 있다.
- **H2(robustness 유지)는 확인했다.** Gate 모델은 Distractor TwoRoom N=2에서 90~94%로, 같은 data로 학습한 CF-JEPA(84%)보다 낮지 않다. Contact 보조 loss와 gate를 붙여도 distractor robustness가 깨지지 않는다. 다만 seed 1개, 50 episodes다.

**남은 일**
- Obstacle이 성공을 좌우하도록 Collision env를 어렵게 만들고(교수님 질문인 obstacle 개수 늘리기와 연결), 같은 모델로 다시 평가한다.
- h_int가 contact 효과를 실제로 배우게 하는 방법: distractor 없는 encoder(E10 `n0`)로 학습한 상한 실험, h_int 입력에 obstacle 위치를 직접 주는 oracle 실험, 여러 step 예측 loss.
- Seed 반복과 episode 수 늘리기. 학습에 쓰지 않은 평가 data.

**결과물:** Proton Drive `results/E13_gate_planning/`에 평가 log 21개와 `summary.txt`가 있다. Colab VM은 평가 직후 종료했다.

## 2026-11-03 · E14. 어려운 Collision env 조건에서 기존 모델 평가 (재학습 없음)

**목적:** E13에서 obstacle 1개 Collision env는 obstacle을 모르는 CF-JEPA도 91%를 성공해, gate 효과(H1)를 볼 여지가 없었다. 학습 없이 평가 조건만 어렵게 해서 두 가지를 본다. 첫째, CF-JEPA가 무너지는가(독립 가정이 깨지는 상황의 정량화). 둘째, gate 모델이 버티는가. 교수님의 "obstacle이 많아지면?" 질문에 대한 첫 실험이기도 하다.

**설정**
- 모델: E9 `cfjepa_collision_n2`(epoch 10), E12 `e12_sup`(gate learned / off), `e12_gain`(epoch 8). 모두 obstacle 1개 data로 학습했다.
- 조건(`--config-name=collision_tworoom`, 50 episodes × seed 42, 43):
  - `c3`: `world.num_obstacles=3`
  - `c3b`: `c3` + `eval.eval_budget=30`(기본 50)
  - `c2f`: `world.num_obstacles=2 +world.obstacle_speed=[6.0,10.0]`(기본 3~6px/step)
- 시작 상태와 goal은 obstacle 1개 data(2,000 episode)에서 뽑는다. Obstacle은 env가 reset 때 agent→goal 경로를 가로지르게 놓는다(`spawn_mode=crossing`).
- 수정: 평가 pipeline이 info buffer를 dataset 모양(obstacle 1개)으로 잡아서, obstacle이 여러 개면 shape 오류가 났다. `envs/collision_tworoom.py`에 `obstacle_info='first'`(info에 첫 obstacle만) 옵션을 더해 `+world.obstacle_info=first`로 평가한다.

**결과** (success rate %, seed 2개 평균. 기본 조건은 E13의 seed 3개 평균)

| 조건 | CF-JEPA | `e12_sup` learned | `e12_sup` off | `e12_gain` |
|---|---|---|---|---|
| 기본 (obstacle 1개) | 91.3 | 91.3 | 84.7 | 85.3 |
| `c3` obstacle 3개 | 73.0 | 71.0 | 75.0 | 69.0 |
| `c3b` 3개 + budget 30 | **65.0** | 66.0 | 69.0 | 67.0 |
| `c2f` 2개 + 빠름 | 73.0 | 82.0 | 82.0 | 82.0 |

**해석**
- **Obstacle이 많아지면 CF-JEPA가 떨어진다**(91% → 73%, 시간 제한까지 두면 65%). CF-JEPA가 §V에 적은 독립 가정의 한계를 처음으로 수치로 보였다. 다만 obstacle 1개 data로 학습한 모델이라 분포 밖 조건이라는 점도 섞여 있다.
- **Gate는 차이를 만들지 못했다.** 모든 조건에서 gate learned와 off가 noise 범위(100 episodes에서 표준오차 약 ±4.5%p) 안에 있다. E12에서 h_int의 예측 이득이 0.5%였던 것과 맞는다.
- `c2f`에서 E12 모델 셋이 모두 82%로 CF-JEPA(73%)보다 높다. Gate를 꺼도 같으므로 gate 효과가 아니다. contact 보조 loss를 붙인 encoder 쪽 차이로 보이지만, seed 2개라 확정할 수 없다.

**다음 (E15):** obstacle 3개 data로 학습해 contact가 잦게 하고, distractor 없이 학습해 encoder가 obstacle을 정밀하게 담게 한다(E10 `n0`에서 oracle 대비 99%). 이 조건에서 contact 지도 gate와 CF-JEPA를 다시 비교한다.

**결과물:** Proton Drive `results/E14_hard_collision/`에 평가 log와 `status.txt`가 있다. 실행 script는 `outputs/e14/e14_pipeline.sh`(git 제외)이다.

## 2026-11-03 · E15. obstacle 3개 data, distractor 없이 학습한 CF-JEPA와 contact 지도 gate

**목적:** E10에서 distractor 없이 학습하면 latent가 obstacle 정보를 oracle만큼 담았다(oracle 대비 99%). E14에서는 obstacle이 많을수록 CF-JEPA가 떨어졌다. 그래서 다음 조건에서 gate가 효과를 내는지 본다. 학습은 obstacle 3개 data로 해서 contact가 잦게 하고, encoder가 obstacle 정보를 담을 수 있게 distractor 없이 한다.

**Data**
- `python scripts/collect_collision.py --episodes 5000 --num-obstacles 3 --name collision3` (약 34분)
- `add_contact_labels.py --data collision3_contact.h5`
- 통계: contact step 비율 16.5%(E4는 7.3%), contact_next 31.1%, contact가 한 번 이상 있는 episode 92.5%

**학습** (Colab A100 한 장에 두 학습, 10 epoch, epoch당 약 23분, 실행 script는 `outputs/e15/e15_pipeline.sh`)
- 공통: `data=collision_tworoom_contact data.dataset.name=collision3_contact.h5 distractor.num=0 trainer.max_epochs=10`
- `e15_cf`: `model=cfjepa`
- `e15_sup`: `model=cfjepa_gate model.predictor.gate_objective=supervised` (contact 보조 loss 없음)
- 세션 손실에 대비해 epoch마다 전체 checkpoint(optimizer 상태 포함)를 로컬로 받았다(`outputs/e15/sync.sh`). 이번에는 세션이 끊기지 않았다.

**Planning 평가** (`--config-name=collision_tworoom +world.obstacle_info=first eval.dataset_name=collision3 world.num_obstacles=3`, 50 episodes × seed 42/43/44)

| 모델 | obstacle 3개 | 평균 | 3개 + budget 30 | 평균 |
|---|---|---|---|---|
| `e15_cf` (CF-JEPA) | 76 / 82 / 76 | 78.0 | 60 / 60 / 62 | 60.7 |
| `e15_sup` gate learned | 84 / 78 / 80 | **80.7** | 66 / 62 / 50 | 59.3 |
| `e15_sup` gate off | 74 / 66 / 80 | 73.3 | 66 / 56 / 58 | 60.0 |
| `e15_sup` gate always (α=1, ablation) | 82 / 68 / 78 | 76.0 | 68 / 66 / 58 | 64.0 |

- 평가 data의 시작 상태는 학습 data(seed 3072)에서 뽑으므로 학습에 쓴 episode와 겹친다.
- 평가 전에 `obstacle_info=first`와 `all`을 epoch 7 checkpoint로 2 episode씩 시험했고, 둘 다 오류 없이 돌았다.

**Gate 분석** (`analyze_gate.py --data collision3.h5 --episodes 500`, distractor 없음)

| | `e15_sup` | E12 `e12_sup` (참고) |
|---|---|---|
| α 평균 non-contact / contact | 0.180 / 0.747 | 0.108 / 0.518 |
| AUROC(α → contact) | **0.937** | 0.905 |
| h_int 상대 이득 non-contact / contact | 0.51% / **0.68%** | 0.39% / 0.49% |
| z^c 오차 contact (CF-JEPA `e15_cf`) | 0.173 (0.160) | 0.082 |

**잔차 probe** (`probe_residual.py --data collision3.h5 --episodes 5000`, contact 구간 B)

| 모델 | zc | zc+zu | oracle |
|---|---|---|---|
| `e15_sup` | 0.051 | 0.062 | 0.043 |
| `e15_cf` | 0.022 | 0.027 | 0.018 |

- obstacle 3개 data에서는 oracle(정답 obstacle 위치, 속도)로도 contact 잔차를 2~4%밖에 설명하지 못한다. E9, E10의 obstacle 1개 data(40~50%)와 비교할 수 없다. 원인은 확인하지 않았다. probe 입력이 obstacle 3개와 그 상호작용을 다 담기에 부족하거나, 잔차가 contact 외의 오차로 채워졌을 가능성이 있다.

**해석**
- **H1은 이번에도 확인하지 못했다.** obstacle 3개 조건에서 gate learned(80.7%)가 CF-JEPA(78.0%)보다 2.7%p 높지만, 150 episodes의 표준오차(약 ±3.3%p) 안이다. 시간 제한 조건에서는 차이가 없다.
- gate learned는 같은 모델의 gate off보다 7.3%p 높다. 하지만 α=1로 항상 켜도 76.0%이고, h_int의 예측 이득은 0.68%뿐이다. 그래서 E13에서처럼 h_int가 충돌 효과를 맞혀서 생긴 차이라기보다, g와 h_int를 함께 학습해서 생긴 의존성일 가능성이 크다.
- **distractor를 없애도 h_int 이득은 0.5%에서 0.68%로 거의 그대로다.** 3 epoch probe(E10)에서 본 "distractor 없으면 latent에 obstacle 정보가 남는다"는 결과가 h_int의 이득으로 이어지지 않았다. 이제 병목은 encoder만이 아니다. h_int의 학습 신호와 용량, 또는 충돌 효과 자체를 예측하기 어려운 점을 의심한다. obstacle 3개 data에서는 oracle도 contact 잔차를 거의 설명하지 못했다.
- gate의 충돌 판별은 더 좋아졌다(AUROC 0.937).

**남은 일**
- h_int의 예측 이득이 왜 작은지 따로 본다. 후보: oracle state를 h_int에 직접 넣는 상한 실험, h_int 용량 키우기, 여러 step 예측 loss.
- obstacle 1개 data에서 distractor 없이 10 epoch 학습한 gate(E10 `n0`의 10 epoch 버전)로 같은 분석을 한다.

**결과물:** Proton Drive `results/E15_obstacle3_nodistractor/`에 두 모델의 `weights_epoch_10.pt`와 config, 평가 log, 분석 txt와 csv가 있다. Colab VM은 분석 직후 종료했다.

## 2026-11-04 · E16. h_int에 정답 state를 넣은 상한 실험 (oracle h_int)

**목적:** E15까지 h_int의 예측 이득이 1% 아래였다. 원인이 "latent에서 obstacle 정보를 꺼내지 못해서"인지, "h_int 자체가 충돌 효과를 배우지 못해서"인지 나눈다. h_int 입력에 정답 state를 직접 넣고 이득이 커지는지 본다.

**설정**
- 코드: `InteractionGatedPredictor`에 `int_oracle_dim` 추가. h_int 입력이 `[z^c, z^u, oracle]`이 된다. oracle은 frame마다 agent 위치(2), obstacle 위치(2), 속도(2), obstacle − agent(2)의 8차원이고 z-score 정규화한다(`scripts/add_oracle_state.py`, `configs/train/data/collision_tworoom_oracle.yaml`). planning에서는 미래 state를 모르므로 진단 전용이다.
- data: Collision TwoRoom obstacle 1개, 1만 episode 새로 수집. distractor N=2.
- 두 모델 모두 supervised gate(E9), 4 epoch, A100에서 동시에 학습했다(epoch당 약 30분).
  - `e16_oracle`: `+model.predictor.int_oracle_dim=8`
  - `e16_base`: 같은 설정, oracle 없음
- 분석: `analyze_gate.py --episodes 500 --distractors 2` (oracle 모델은 oracle을 함께 넣는다).
- 학습을 마친 세션이 분석 전에 끊겼다. pipeline의 `wait`가 메모리 기록용 반복 작업까지 기다려서 분석이 시작되지 않은 사이였다. epoch별 weights는 받아 두었으므로 새 세션(E19)에서 data를 다시 모아 분석했다.

**결과**

| | `e16_oracle` | `e16_base` |
|---|---|---|
| h_int 상대 이득 non-contact | 0.56% | 0.88% |
| h_int 상대 이득 **contact** | **1.43%** | 0.72% |
| z^c 오차 contact (h_int 끔 / 켬) | 0.0439 / 0.0432 | 0.0588 / 0.0584 |
| z^c 오차 non-contact | 0.0175 | 0.0176 |
| α 평균 non-contact / contact | 0.124 / 0.291 | 0.140 / 0.375 |
| AUROC(α → contact) | 0.832 | 0.870 |

- 학습 중 batch 평균 이득도 두 모델 모두 0.2~0.8%였다(oracle 모델은 첫 epoch에만 2~5%까지 올랐다가 내려왔다).

**해석**
- **정답 state를 줘도 충돌 구간 이득은 1.4%다.** E9에서 같은 종류의 정답 state로 MLP probe가 contact 잔차를 40~50% 설명한 것과 차이가 크다. 그래서 병목은 "정보가 없어서"보다 "h_int가 그 정보로 잔차를 맞히도록 학습되지 않아서" 쪽이다.
- 의심하는 원인: lr 5e-5는 zero-init으로 새로 붙인 h_int에 너무 작다(probe는 따로 수렴할 때까지 학습했다). E19에서 h_int와 gate만 lr 1e-3으로 학습해 확인한다.
- oracle 모델은 h_int를 끈 z^c 오차도 contact 구간에서 더 낮다(0.044 대 0.059). 두 모델이 encoder부터 따로 학습되어 생긴 차이일 수 있어, seed 반복 없이는 해석하지 않는다.

**결과물:** Proton Drive `results/E16_oracle_hint/`에 두 모델의 `weights_epoch_4.pt`와 config가 있다. 분석 txt와 csv도 같은 폴더에 있다.

## 2026-11-04 · E17. obstacle 개수(1~5개)에 따른 planning 성공률 (강연우)

**목적:** 프로포절 때 받은 "obstacle이 많아지면 어떻게 되는가" 질문에 답한다. obstacle 1개로 학습한 모델을 obstacle 1~5개 환경에서 평가한다.

**설정**
- 모델: E12 `e12_sup`(gate) epoch 8과 같은 조건의 CF-JEPA. 둘 다 obstacle 1개 data로 학습했다.
- 2, 4, 5개는 새로 평가했다(seed 42, 43, seed당 50 episodes). 1개는 E13, 3개는 E14 값을 그대로 쓴다.
- 코드는 `yeonwoo/midterm-figs` branch의 `scripts/colab/eval_obstacle_sweep.sh`, `scripts/plot_obstacle_sweep.py`다(PR #1로 main에 합쳤다).

**결과** (성공률 %, 표준오차 약 ±4~5%p)

| obstacle 수 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| CF-JEPA | 91.3 | 78.0 | 73.0 | 73.0 | 66.0 |
| gate | 91.3 | 77.0 | 71.0 | 68.0 | 68.0 |

- seed별 값: 2개 CF-JEPA 80 / 76, gate 80 / 74. 4개 CF-JEPA 72 / 74, gate 72 / 64. 5개 CF-JEPA 70 / 62, gate 68 / 68.

**해석**
- obstacle이 늘수록 두 모델 모두 떨어진다(1개에서 5개로 약 −25%p). CF-JEPA의 독립 가정이 obstacle 수가 늘 때 성능 저하로 나타난다.
- 두 모델의 차이는 모든 점에서 오차 범위 안이다. 같은 조건에서도 seed에 따라 최대 8%p 차이가 난다.
- 학습은 obstacle 1개로만 했으므로, 하락에는 "학습 때 못 본 상황"의 영향도 섞여 있다. 이를 나누려면 여러 obstacle data로 학습해야 한다(E15가 3개로 학습한 경우다).

**결과물:** Proton Drive `발표자료/midterm_figs/`의 `obstacle_sweep.png`, `.svg`, `.csv`, `obstacle_sweep_summary.txt`. 중간 발표 덱 6페이지에 넣었다.

## 2026-11-04 · E18. Planning 중 gate α_t 시각화 (강연우)

**목적:** 실제 planning에서 gate가 충돌 순간에 켜지는지 영상으로 확인한다(demo).

**설정**
- 모델: E12 `e12_sup` epoch 8, Collision TwoRoom obstacle 1개, seed 42, 16 episodes.
- 코드는 `yeonwoo/midterm-figs` branch의 `scripts/record_gate_rollout.py`다(PR #1로 main에 합쳤다). 화면 아래에 α_t 선과 실제 contact 구간(띠)을 함께 그린다.

**결과**
- 성공률 87.5%(16 episodes).
- α가 계산된 197 step에서 contact_next 비율 12.2%. α 평균 non-contact / contact = 0.156 / 0.494, AUROC 0.874.
- episode 2: obstacle과 세 번 충돌하고, 충돌마다 α가 0.70~0.79로 올라갔다가 멀어지면 0.03~0.09로 내려간다. 발표 영상으로 쓴다.
- episode 15: 충돌이 끝난 뒤에도 agent가 바로 옆에 있는 3 step 동안 α가 0.66~0.77로 남는다. 실제 접촉보다 "아주 가까움"에 반응하는 것으로 보인다.
- episode 0은 충돌 직후에 α가 켜져 타이밍이 늦고, episode 5는 충돌이 α 계산 시작(t=10) 전에 일어나 확인할 수 없다.

**해석:** E12 분석(AUROC 0.905)과 같은 결론이다. gate는 충돌 순간을 알아차리지만, 켜졌을 때의 h_int 예측 이득은 0.5% 수준이라 성공률은 CF-JEPA와 같다.

**결과물:** Proton Drive `발표자료/midterm_figs/`의 `gate_env{0,2,5,15}.gif`, `.png`, `steps.csv`, `summary.txt`, `arrange.txt`. 중간 발표 덱 8페이지에 넣었다.

## 2026-11-04 · E19. h_int와 gate만 lr 1e-3으로 학습

**목적:** E16에서 h_int가 정답 state를 받고도 배우지 못했다. h_int, `int_in`, `int_out`, gate만 lr을 1e-3으로 올리고(나머지는 5e-5) 이득이 커지는지 본다.

**설정**
- `scripts/train.py`에 `model.int_lr`을 추가했다. 주면 stable-pretraining의 regex 기반 optimizer 분리로 `model.predictor.(h_int|int_in|int_out|gate)`에 따로 AdamW를 붙인다. smoke test에서 `int_opt` 73만 개, `model_opt` 1905만 개 parameter로 나뉜 것을 확인했다.
- data와 나머지 설정은 E16과 같다(obstacle 1개, 1만 episode, distractor N=2, supervised gate, 4 epoch). E16 세션이 끊겨 새 세션에서 data를 다시 모았다.
  - `e19_lr`: `+model.int_lr=1e-3`
  - `e19_lr_oracle`: 위 + `+model.predictor.int_oracle_dim=8`

**결과** (`analyze_gate.py --episodes 500 --distractors 2`)

| | `e16_base` (lr 5e-5) | **`e19_lr`** | `e16_oracle` (lr 5e-5) | `e19_lr_oracle` |
|---|---|---|---|---|
| h_int 상대 이득 non-contact | 0.88% | 6.6% | 0.56% | 13.6% |
| h_int 상대 이득 **contact** | 0.72% | **5.2%** | 1.43% | **37.9%** |
| z^c 오차 contact (h_int 끔 → 켬) | 0.0588 → 0.0584 | 0.0647 → **0.0582** | 0.0439 → 0.0432 | 0.0653 → 0.0368 |
| α 평균 non-contact / contact | 0.140 / 0.375 | 0.124 / 0.564 | 0.124 / 0.291 | 0.111 / 0.515 |
| AUROC(α → contact) | 0.870 | **0.911** | 0.832 | 0.904 |

**해석**
- **lr만 바꿔서 h_int가 처음으로 의미 있게 배웠다.** 정답 state 없이 실제로 쓸 수 있는 모델(`e19_lr`)에서 contact 구간 z^c 오차가 h_int로 10% 줄었다(0.0647 → 0.0582). E8~E16의 1% 미만과 비교된다. 정답 state를 주면 44% 준다. 그래서 E16, E20의 진단대로 병목은 h_int의 학습 속도였다.
- 다만 `e19_lr`은 h_int를 끈 g의 contact 오차가 E16보다 높다(0.0647 대 0.0588). h_int가 일을 나눠 가지면서 g가 충돌 구간을 덜 맞히게 됐을 수 있다. 그래서 h_int를 켠 최종 오차는 E16 base와 거의 같다(0.0582 대 0.0584). 성공률이 오르는지는 E21에서 본다.
- non-contact에서도 6.6% 이득이 난다. gate가 non-contact에서는 α 0.12로 닫혀 있으므로 실제 예측에는 작게 반영된다.
- 정답 state와의 차이(5% 대 38%)는 latent의 obstacle 정보가 부족하다는 E10, E20의 결론과 맞다. 다음 개선 대상은 encoder다.

**결과물:** Proton Drive `results/E19_hint_lr/`에 두 모델의 `weights_epoch_4.pt`와 config, 분석 txt와 csv가 있다.

## 2026-11-04 · E20. 학습된 모델을 고정하고 h_int만 새로 학습 (frozen h_int fit)

**목적:** E16에서 정답 state를 넣어도 h_int 이득이 1.4%였다. 원인이 h_int 구조인지, 함께 학습하는 방식인지, 잔차 자체가 어려운지 나눈다.

**설정**
- `scripts/fit_hint_frozen.py`. E16 모델(`e16_oracle`, `e16_base` epoch 4)을 고정하고, clip마다 z^c, z^u, action embedding, g의 예측(base), 다음 frame z^c, 정답 state, contact 라벨을 뽑는다(학습 3000 episode, 약 3만 clip / test 마지막 500 episode, 5133 clip, distractor N=2).
- 새 head를 잔차(target − base)에 AdamW로 4000 step(batch 512) 학습한다. 출력은 학습과 같은 `tanh` 형태다.
  - `tf_*`: 모델의 h_int와 같은 구조(AdaLN-zero transformer, depth 2, 출력층 zero-init)
  - `mlp_*`: frame별 3층 MLP
  - 입력: `oracle`(z^c, z^u, 정답 state), `latent`(z^c, z^u), `oracle_only`(정답 state, action)
- E19 세션에서 학습과 같이 돌렸다.

**결과** (test 상대 이득, non-contact / contact)

| 모델 | head | lr | non-contact | contact |
|---|---|---|---|---|
| `e16_oracle` | 모델 안의 h_int (E16) | 5e-5, 함께 학습 | 0.60% | 1.41% |
| `e16_oracle` | tf_oracle | 1e-3 | 16.4% | **26.1%** |
| `e16_oracle` | tf_oracle | 5e-5 | 3.4% | **22.5%** |
| `e16_oracle` | mlp_oracle_only | 1e-3 | 12.0% | 26.5% |
| `e16_oracle` | tf_latent | 1e-3 | 8.3% | −10.9% |
| `e16_oracle` | mlp_latent | 1e-3 / 5e-5 | −15.2% / 7.2% | −30.3% / 3.5% |
| `e16_base` | 모델 안의 h_int (E16) | 5e-5, 함께 학습 | 0.89% | 0.79% |
| `e16_base` | tf_oracle | 1e-3 | 12.1% | **29.3%** |
| `e16_base` | mlp_oracle_only | 1e-3 | 5.2% | 23.9% |
| `e16_base` | tf_latent | 1e-3 | 3.8% | −15.8% |

**해석**
- **h_int 구조는 문제가 아니다.** 같은 구조의 새 h_int가 정답 state로 contact 잔차를 26~29% 줄인다. lr 5e-5로도 22.5%다.
- **같은 lr인데 함께 학습하면 1.4%, 고정하고 학습하면 22.5%다.** 그래서 병목은 함께 학습하는 방식이다. encoder, g, target이 같이 움직이는 동안 작은 lr의 h_int가 잔차를 따라가지 못한 것으로 본다. E19에서 h_int만 lr 1e-3으로 올리자 함께 학습해도 이득이 커졌다(학습 중 batch 평균 oracle 모델 8~13%, oracle 없는 모델 3~4.5%).
- **latent만 넣으면 고정 모델에서는 contact 구간이 오히려 나빠진다**(tf_latent −11~−16%). 학습 data에는 맞추지만 test로 일반화하지 못한다. 이 encoder의 latent(distractor 있는 학습)에는 contact 잔차를 일반화해 설명할 obstacle 정보가 부족하다는 E10의 진단과 같다.
- 정리: 1) lr을 키우면 h_int가 배울 수 있다(E19). 2) 실제 모델에서 남는 병목은 latent의 obstacle 정보다.

**결과물:** `outputs/e20/*.log`를 Proton Drive `results/E20_frozen_hint_fit/`에 올렸다.

## 2026-11-04 · E21. lr을 키운 gate(E19)의 planning 성공률

**목적:** E19에서 h_int가 contact 구간 z^c 오차를 10% 줄였다. 이것이 planning 성공률로 이어지는지 본다.

**설정**
- `python scripts/eval.py --config-name=collision_tworoom +world.obstacle_info=first policy=<ckpt> seed=<42|43|44> [조건]`, CEM 300×30, seed당 50 episodes(조건당 150).
- 조건: obstacle 1개(기본) / obstacle 3개 + `eval.eval_budget=30`(E14의 가장 어려운 조건).
- 모델: CF-JEPA(E9 `cfjepa_collision_n2`, 10 epoch), `e19_lr`(gate learned / off), `e16_base`(lr 5e-5 gate, 같은 4 epoch).
- E19 세션에서 E22 학습과 동시에 돌렸다. pipeline의 결과 파일 이름은 변수 덮어쓰기 때문에 `c3b_c3b_c1_...`처럼 길어졌지만 순서(c1 s42, 43, 44, c3b s42, 43, 44)는 맞다.

**결과** (success rate %)

| 모델 | 1개 s42 / 43 / 44 | **1개 평균** | 3개+30 s42 / 43 / 44 | **3개+30 평균** |
|---|---|---|---|---|
| CF-JEPA (10 epoch) | 90 / 90 / 84 | **88.0** | 58 / 66 / 56 | **60.0** |
| `e19_lr` gate learned | 92 / 94 / 78 | **88.0** | 66 / 68 / 56 | **63.3** |
| `e19_lr` gate off | 82 / 86 / 80 | 82.7 | 68 / 72 / 66 | 68.7 |
| `e16_base` gate learned | 88 / 86 / 80 | 84.7 | 64 / 76 / 62 | 67.3 |

**해석**
- **H1은 이번에도 확인하지 못했다.** 예측이 좋아진 `e19_lr`도 1개 조건에서 CF-JEPA와 같고(88.0), 어려운 조건에서 3.3%p 높지만 표준오차(약 ±4%p) 안이다.
- 어려운 조건에서는 오히려 같은 모델의 gate off(68.7)와 `e16_base`(67.3)가 더 높다. 1개 조건에서는 반대로 gate off가 낮다. 조건마다 순위가 바뀌므로 noise 수준의 차이로 본다.
- 예측 개선(contact z^c 오차 −10%)이 성공률로 이어지지 않았다. 후보 원인: 1) 개선 폭이 아직 작다(정답 state 상한은 −44%). 2) planning cost는 goal까지의 z^c 거리라, 충돌 직후 몇 step의 예측 오차가 계획 결과를 크게 바꾸지 않는다. 3) 1만 episode, 4 epoch 모델이라 CF-JEPA(10 epoch)보다 덜 학습됐다.
- 2607.10362(planner가 도달하는 state의 예측 충실도가 성공을 좌우하고 data 평균 오차와는 거의 무관하다)와 같은 방향의 관찰이다.

**결과물:** `outputs/e21/res/status.txt`를 Proton Drive `results/E21_hint_lr_planning/`에 올렸다.

## 2026-11-04 · E22. gate + h_int lr 1e-3 + contact 보조 loss, 같은 4 epoch의 CF-JEPA

**목적:** E19에서 h_int가 배우기 시작했고, 남은 병목은 latent의 obstacle 정보다(E20). encoder에 contact 보조 loss(E10, E12, 라벨 2%)를 더해 h_int 이득이 커지는지 본다. E21의 CF-JEPA는 10 epoch이라, 같은 data와 4 epoch로 맞춘 CF-JEPA도 함께 학습한다.

**설정** (E19 세션, E21 평가와 동시에 학습)
- `e22_aux`: `model=cfjepa_gate data=collision_tworoom_aux model.predictor.gate_objective=supervised +model.int_lr=1e-3 +model.contact_head=true +model.contact_aux_weight=1.0 +model.contact_aux_key=contact_next_p2`, 4 epoch
- `e22_cf`: `model=cfjepa data=collision_tworoom_contact`, 4 epoch
- 라벨 2% column은 `add_contact_labels.py --label-frac 0.02`로 만들었다(190 / 10000 episode).

**결과** (`analyze_gate.py --episodes 500 --distractors 2`)

| | `e22_aux` | `e19_lr` (참고) | `e22_cf` |
|---|---|---|---|
| h_int 상대 이득 non-contact / contact | 10.1% / **7.0%** | 6.6% / 5.2% | |
| contact z^c 오차 (h_int 끔 → 켬) | 0.0883 → 0.0800 (−9.3%) | 0.0647 → 0.0582 (−10%) | 0.0511 |
| α 평균 non-contact / contact | 0.117 / 0.513 | 0.124 / 0.564 | |
| AUROC(α → contact) | 0.906 | 0.911 | |

- `e22_aux`는 z^c 오차의 절댓값이 다른 모델보다 크다(non-contact 0.033 대 0.018). 보조 loss가 latent의 분포를 바꾸므로 모델 사이 절댓값은 비교하지 않는다.

**해석:** 보조 loss를 더하면 contact 상대 이득이 5.2%에서 7.0%로 조금 커지지만, contact 구간 오차의 감소율은 E19와 비슷하다(약 10%). 보조 loss의 효과는 작다.

**결과물:** Proton Drive `results/E22_aux_lr/`에 두 모델의 `weights_epoch_4.pt`, config, 분석 txt와 csv가 있다.

## 2026-11-04 · E23. E22 모델의 planning 성공률

**설정:** E21과 같다(obstacle 1개 / 3개 + budget 30, seed 42, 43, 44, 조건당 150 episodes). 모델은 `e22_cf`(4 epoch CF-JEPA), `e22_aux` gate learned, `e22_aux` gate off.

**결과** (success rate %)

| 모델 | 1개 s42 / 43 / 44 | **1개 평균** | 3개+30 s42 / 43 / 44 | **3개+30 평균** |
|---|---|---|---|---|
| `e22_cf` CF-JEPA (4 epoch) | 86 / 92 / 84 | **87.3** | 76 / 72 / 56 | **68.0** |
| `e22_aux` gate learned | 82 / 94 / 86 | **87.3** | 60 / 58 / 54 | **57.3** |
| `e22_aux` gate off | 84 / 86 / 82 | 84.0 | 62 / 68 / 54 | 61.3 |

**해석**
- obstacle 1개에서는 gate 모델과 CF-JEPA가 같다(87.3).
- 어려운 조건에서는 gate 모델이 CF-JEPA보다 10.7%p 낮다. 표준오차(약 ±4%p)보다 크다. 같은 모델의 gate off(61.3)도 CF-JEPA보다 낮으므로, gate보다는 보조 loss가 encoder를 바꾼 영향으로 본다. E12(보조 loss + gate, 10 epoch)도 어려운 조건(E14)에서 CF-JEPA와 같은 수준이었으므로, 4 epoch에서 보조 loss가 아직 다른 학습을 방해하고 있을 수 있다.
- E19~E23을 합치면: h_int의 예측 이득은 lr 수정으로 7배 이상 커졌지만(0.7% → 5~7%), planning 성공률은 어떤 조건에서도 CF-JEPA보다 높지 않았다.

**결과물:** `outputs/e23/res/status.txt`를 Proton Drive `results/E23_aux_lr_planning/`에 올렸다. 평가 뒤 Colab 세션을 종료했다(남은 CU 52).

## 2026-11-04 · E24. 조건별 재평가: 충돌이 성공을 좌우하는 조건 고르기 (재학습 없음)

**목적:** 지금까지의 조건에서는 gate 효과를 볼 여지가 있는지 분명하지 않았다. obstacle 없는 대조군을 함께 재서, 충돌 때문에 떨어지는 폭이 가장 큰 조건을 고른다. E25부터 이 조건을 기준으로 삼는다.

**설정**
- 모델: 같은 data와 4 epoch로 학습한 `e22_cf`(CF-JEPA), `e19_lr`(gate learned / off). 모두 obstacle 1개, 반지름 14 data로 학습했다.
- `python scripts/eval.py --config-name=collision_tworoom +world.obstacle_info=first policy=<ckpt> seed=<42|43> [조건]`, seed당 50 episodes(조건당 100).
- 평가 예산의 하한은 25다(planning horizon 5 × action block 5). 그래서 시간 제한은 30으로 둔다.
- `num_obstacles=0` 평가를 위해 `obstacle_info=first`에서 빈 obstacle을 0으로 채우게 했다.

**결과** (success rate %, seed 2개 평균. 표준오차 약 ±4~5%p)

| 조건 | CF-JEPA | gate learned | gate off |
|---|---|---|---|
| `a0` obstacle 없음 | 91 | 93 | 93 |
| `b0` obstacle 없음, budget 30 | 89 | 91 | 91 |
| `c1b` 1개, budget 30 | 82 | 77 | 76 |
| `c5b` 5개, budget 30 | 55 | 57 | 59 |
| `f3b` 3개, 속도 6~10, budget 30 | 66 | 68 | 63 |
| **`r3b` 3개, 반지름 22, budget 30** | **43** | **48** | **44** |

**해석**
- 시간 제한만으로는 거의 떨어지지 않는다(`a0` 91 → `b0` 89). E14, E21의 어려운 조건에서 성공률이 떨어진 원인은 obstacle이다.
- **`r3b`를 기준 조건으로 고른다.** 같은 budget의 대조군보다 CF-JEPA가 46%p 떨어져 하락 폭이 가장 크다.
- gate 차이는 어느 조건에서도 표준오차 안이다. `r3b`의 gate 48 대 CF-JEPA 43도 마찬가지다.
- 한계: 하락 폭이 크다고 해서 충돌을 예측하면 회복할 수 있다는 보장은 없다. 큰 obstacle 3개는 학습 data(반지름 14, 1개)와 달라, 분포 밖 조건의 영향도 섞여 있다.

**운영 메모:** 처음에는 평가 6개와 data 수집을 동시에 돌려 CPU(12개) load가 37까지 올랐고, 평가 하나가 3.5분에서 20분으로 늘었다. GPU 사용률은 0%였으므로 병목은 CPU다. 수집을 SIGSTOP으로 멈추고 평가를 3개씩 돌린 뒤 수집을 이어 갔다. 평가는 한 세션에서 3개씩 돌린다.

**결과물:** `outputs/e24/res/status.txt`를 Proton Drive `results/E24_condition_screen/`에 올렸다.

## 2026-11-04 · E25. 여러 step 앞까지 이어 붙인 예측 loss (최종 발표용)

**목적:** planning은 예측을 되먹여 5 step 앞까지 가는데, 지금까지의 학습은 1 step 예측 오차만 줄였다. 학습에서도 예측을 되먹이는 loss를 더하면 충돌을 포함한 긴 구간 예측과 성공률이 좋아지는지 본다.

**설정**
- 구현: `scripts/train.py`의 `loss.rollout_steps`, `loss.rollout_weight`. history 3 frame에서 k step 앞까지 예측을 되먹이고 각 step의 embedding 오차를 평균해 loss에 더한다. clip 길이를 6으로 늘린 `collision_tworoom_rollout` data config를 쓴다.
- `e25_gate`: `model=cfjepa_gate model.predictor.gate_objective=supervised +model.int_lr=1e-3 data=collision_tworoom_rollout +loss.rollout_steps=3 +loss.rollout_weight=1.0 trainer.max_epochs=4`
- `e25_cf`: `model=cfjepa`, 나머지는 같다.
- 평가: `python scripts/eval.py --config-name=collision_tworoom +world.obstacle_info=first policy=<ckpt> seed=<42|43|44> [조건]`, 조건당 150 episodes. 비교 대상은 rollout loss 없이 같은 data, 같은 4 epoch로 학습한 `e22_cf`, `e19_lr`이다. `r3b`의 비교 대상은 E24의 seed 2개에 이번에 잰 seed 44를 합쳤고, 나머지 조건은 E21, E23의 값이다.
- 긴 구간 예측 진단: `python scripts/multistep_error.py --ckpts <ckpt들>`, test 500 episodes, 3459 clips. 모델마다 latent 크기가 달라 persistence(마지막 관측 z^c를 그대로 쓴 예측) 오차로 나눈 비율을 쓴다. 낮을수록 좋다.

**결과: 성공률** (success rate %. 표준오차 약 ±4%p)

| 조건 | `e22_cf` | `e25_cf` | `e19_lr` learned | `e25_gate` learned | `e19_lr` off | `e25_gate` off |
|---|---|---|---|---|---|---|
| obstacle 1개 | 87.3 | 80.0 | 88.0 | 83.3 | 82.7 | 82.0 |
| 3개, budget 30 | 68.0 | 64.0 | 63.3 | 62.0 | 68.7 | 67.3 |
| `r3b` 3개, 반지름 22, budget 30 | 45.3 | 48.0 | 48.0 | 46.0 | 46.0 | 44.0 |

**결과: 긴 구간 예측** (persistence 대비 오차 비율. contact는 그 구간 안에 충돌이 있는 clip)

| k step 앞 | `e22_cf` 전체 / contact | `e25_cf` 전체 / contact | `e19_lr` 전체 / contact | `e25_gate` 전체 / contact |
|---|---|---|---|---|
| 1 | 0.408 / 0.645 | 0.469 / 0.596 | 0.358 / 0.642 | 0.449 / 0.574 |
| 3 | 0.297 / 0.534 | 0.301 / 0.535 | 0.261 / 0.544 | 0.306 / 0.523 |
| 5 | 0.248 / 0.457 | 0.230 / 0.447 | 0.225 / 0.461 | 0.231 / 0.435 |

**h_int 이득:** `e25_gate`의 contact 구간 이득은 6.67%(`e19_lr` 5.2%), α AUROC는 0.899(`e19_lr` 0.911)다.

**해석**
- **H1은 이번에도 확인하지 못했다.** 기준 조건 `r3b`에서 `e25_cf` 48.0, `e25_gate` 46.0으로 모두 표준오차 안이다. gate가 CF-JEPA보다 높지도 않다.
- 쉬운 조건에서는 rollout loss를 넣은 모델이 오히려 낮다. obstacle 1개에서 CF-JEPA가 7.3%p, gate가 4.7%p 내려갔다. CF-JEPA의 하락은 표준오차의 2배에 못 미쳐 확정할 수는 없지만, 올라간 조건은 없다.
- **rollout loss가 긴 구간 예측 자체를 크게 바꾸지 못했다.** 직접 학습한 k=3의 전체 비율은 CF-JEPA 0.297 → 0.301, gate 0.261 → 0.306으로 나아지지 않았다. contact 구간은 k=1에서 8~11%, k=5에서 2~6% 줄었지만, 충돌 없는 k=1 오차는 CF-JEPA 0.343 → 0.439, gate 0.282 → 0.416으로 늘었다.
- 따라서 이번 결과는 "긴 구간 예측이 좋아져도 성공률이 그대로"라는 뜻이 아니다. 긴 구간 예측이 충분히 좋아지지 않아, 그 가설은 아직 검증되지 않았다.
- 한계: 비율의 분모도 모델마다 다르므로 모델 간 비교는 근사다. clip은 겹치므로 3459개가 독립 표본은 아니다. weight 1.0, 3 step, 4 epoch 한 설정만 봤다.

**남은 문제:** planner가 실제로 도달하는 state로 학습하는 실험은 CU가 충전되는 11/10 이후로 미룬다.

**결과물:** `results/E25_rollout_loss/`에 `e25_gate`, `e25_cf`(`weights_epoch_4.pt`, `config.json`), 분석 결과, `status.txt`, `eval_status.txt`, `multistep.txt`를 올렸다. 평가 뒤 Colab 세션을 종료했다(남은 CU 8.60).

## 2026-11-12 · E26. Planning 입력 고치기: 실제 첫 화면과 과거 frame 3장 (재학습 없음)

**목적:** E13~E25에서 gate가 planning에서 한 번도 CF-JEPA를 넘지 못했다. 그래서 planning이 모델에 무엇을 넣는지 stable-worldmodel 0.1.1 원본을 확인했고, 문제 두 가지를 찾았다.
1. **첫 관측이 실제 화면이 아니다.** `World._evaluate_from_dataset`은 env를 dataset 시작 상태로 맞춘 뒤 `infos['pixels']`를 dataset frame으로 덮어쓴다(world.py 523~530행). 원래 TwoRoom에서는 둘이 같다. 하지만 Collision env는 `_set_goal_state`에서 obstacle을 새로 놓으므로, 첫 계획이 실제와 다른 obstacle 배치(r3b라면 큰 obstacle 3개 대신 dataset의 작은 obstacle 1개)를 보고 세워진다. budget 30에서는 첫 계획(25 step을 한 번에 실행)이 성공을 거의 결정한다.
2. **Planning은 frame 1장만 넣는다.** `PlanConfig.history_len`은 정의만 있고 쓰이지 않는다. 학습은 history 3 frame인데 planning에서는 1장이라, obstacle이 움직이는 방향을 알 수 없다.

**구현** (branch `seolcu/plan-history`)
- `scripts/eval.py`의 `+obs_fix=true`: episode 첫 step의 pixels를 `env.render()`로 바꾼다.
- `+plan_history=3`: frame t−10, t−5, t와 그 사이 실행한 action block 2개(`hist_action`, 학습과 같이 정규화)를 넣는다. 시작 전 frame은 `CollisionTwoRoomEnv.render_past(k)`로 만든다. "agent가 시작 위치에 멈춰서 지켜보았다"고 보고, obstacle만 속도를 뒤집어 k step 되돌린 위치에 그린다. action은 0이다.
- `iawm.cfjepa.history_get_cost`: `hist_action`을 후보 action 앞에 붙여 LeWM.rollout을 부른다. history 3 frame + 후보 5 step을 예측한다.
- smoke test(3 episodes)에서 입력 모양을 확인했다. P2는 pixels (3, 3, 224, 224, 3), hist_action (3, 2, 10)이다.

**설정:** E24와 같은 모델(`e22_cf`, `e19_lr` learned / off, 모두 obstacle 1개·반지름 14·distractor N=2 data로 4 epoch). 조건은 r3b, c1b이고 seed 42, 43, 44, 조건당 150 episodes다. P0는 기존 평가(E24, E25 값), P1은 `+obs_fix=true`, P2는 P1에 `+plan_history=3`을 더한 것이다. 평가 data(2,000 episode)는 새로 모았다. A100 약 1시간 10분.

**결과** (success rate %, seed 3개 평균. 표준오차 약 ±4%p)

| 조건 | 프로토콜 | CF-JEPA | gate learned | gate off |
|---|---|---|---|---|
| r3b | P0 (E24, E25) | 45.3 | 48.0 | 46.0 |
| r3b | P1 실제 첫 화면 | **56.7** (54 / 60 / 56) | 51.3 (58 / 44 / 52) | 48.0 (46 / 48 / 50) |
| r3b | P2 + history 3 | 54.0 (62 / 54 / 46) | 50.7 (50 / 52 / 50) | 49.3 (50 / 50 / 48) |
| c1b | P0 (E24, seed 2개) | 82 | 77 | 76 |
| c1b | P1 | 81.3 (82 / 80 / 82) | 82.7 (88 / 74 / 86) | 80.7 (84 / 82 / 76) |
| c1b | P2 | 84.7 (86 / 76 / 92) | 82.7 (78 / 86 / 84) | 80.7 (80 / 80 / 82) |

**해석**
- **첫 화면 버그는 성공률에 크게 영향을 준다.** 실제 화면을 넣으면 r3b에서 CF-JEPA가 45.3에서 56.7로 11.4%p 오른다(표준오차의 약 2.8배). 지금까지의 Collision env 평가는 모두 다른 obstacle 배치를 보고 첫 계획을 세운 것이다. 앞으로 Collision env 평가는 P1 이상으로 한다.
- **History를 넣어도 이 gate 모델은 이득을 보지 못했다.** r3b P2에서 gate 50.7 대 CF-JEPA 54.0이다. 이 모델은 obstacle 1개·작은 크기·distractor 있는 data로 학습해 r3b가 분포 밖이고, E20의 진단대로 latent에 obstacle 정보가 부족하다. 입력을 고친 다음 남은 병목은 모델 쪽이다.
- CF-JEPA는 g가 z^u를 보지 않으므로 history가 obstacle 예측에 쓰이지 않는다. P1과 P2의 차이(56.7 대 54.0)는 noise 범위다.

**다음 (E27):** r3b와 같은 data(obstacle 3개, 반지름 22)로, distractor 없이(E10에서 obstacle 정보 99%), h_int lr 1e-3, 10 epoch로 CF-JEPA와 gate를 학습하고 P2로 평가한다.

**결과물:** `outputs/e26/res/e26/`(git 제외)에 평가 log 39개와 `status.txt`가 있다.

## 2026-11-12 · E27. 평가 조건과 같은 data(obstacle 3개, 반지름 22)로 학습한 CF-JEPA와 gate

**목적:** E26에서 planning 입력을 고친 뒤에도 기존 gate 모델은 이득이 없었다. 원인 후보는 학습 data와 평가 조건(r3b)이 다른 점과, distractor 때문에 latent에 obstacle 정보가 부족한 점(E10)이다. 둘을 함께 없애고 다시 비교한다.

**설정**
- 학습 data: `collect_collision.py --episodes 10000 --num-obstacles 3 --obstacle-radius 22 --name r3`, contact 라벨 추가(`r3_contact.h5`). 76.9만 frame이고 contact step 26.2%, contact가 있는 episode 97.6%, scripted policy의 goal 도달 52.9%다.
- 모델 (distractor 없음, 10 epoch, A100에서 동시 학습 약 9시간):
  - `e27_cf`: `model=cfjepa`
  - `e27_gate`: `model=cfjepa_gate model.predictor.gate_objective=supervised +model.int_lr=1e-3`
- 평가: E26의 P2(`+obs_fix=true +plan_history=3`), r3b와 c1b, seed 42/43/44, 조건당 150 episodes. gate는 learned와 `+gate_mode=off`로 잰다.
- 분석: `analyze_gate.py --episodes 500 --distractors 0` (r3 data 500 episode, 1-step 예측).

**결과: planning** (success rate %, seed 3개 평균. 괄호는 seed별. 표준오차 약 ±4%p)

| 조건 | CF-JEPA | gate learned | gate off | 참고: E26 CF-JEPA (P2) |
|---|---|---|---|---|
| r3b | 49.3 (48 / 50 / 50) | 50.7 (50 / 54 / 48) | 50.7 (52 / 50 / 50) | 54.0 |
| c1b | 73.3 (68 / 76 / 76) | 76.0 (78 / 76 / 74) | 73.3 (78 / 64 / 78) | 84.7 |

**결과: 1-step 예측** (`e27_gate`, contact 비율 46.7%)

| | non-contact | contact |
|---|---|---|
| z^c 오차, h_int 끔 | 0.0428 | 0.1519 |
| z^c 오차, h_int 켬 (α=1) | 0.0435 | **0.1237** (−18.5%) |
| 평균 α | 0.15 | 0.83 |

- AUROC(α → contact) = 0.963이다.
- contact 구간 z^c 오차는 h_int를 켜도 non-contact의 2.9배다.

**해석**
- **예측은 처음으로 분명히 좋아졌다.** contact 구간 z^c 오차가 h_int로 18.5% 줄었다. E15~E22의 0.5~10%보다 크다. 평가 조건과 같은 data, distractor 없음, h_int lr 1e-3의 조합에서 h_int가 충돌 효과를 배운다.
- **그런데 planning 성공률은 그대로다.** r3b에서 gate 50.7, gate off 50.7, CF-JEPA 49.3으로 모두 표준오차 안이다. 1-step 예측 개선이 성공으로 이어지지 않는 현상(E21, E23)이 다시 나왔다.
- obstacle 3개 data로 학습한 두 모델 모두 c1b에서는 E26 모델(obstacle 1개 data)보다 낮다. r3b에서도 E26 CF-JEPA(54.0)보다 높지 않다. data를 평가 조건에 맞춘 것만으로는 성공률이 오르지 않는다.
- 남은 질문은 "충돌을 정확히 예측하면 이 조건에서 성공률을 얼마나 올릴 수 있는가"이다. 상한이 낮다면 모델을 고쳐도 차이가 날 수 없다. E28에서 정답 simulator로 계획해 이 상한을 잰다.

**결과물:** checkpoint `checkpoints/e27_cf`, `checkpoints/e27_gate`(git 제외, epoch 1~10), 평가와 분석 log `outputs/e27/res/e27/`(git 제외).

## 2026-11-12 · E28. 정답 simulator로 계획한 상한 (학습 없음, CPU)

**목적:** E27에서 1-step 충돌 예측은 좋아졌지만 성공률은 그대로였다. 충돌을 정확히 예측하면 이 평가 조건에서 성공률을 얼마나 올릴 수 있는지(상한)를 먼저 잰다. 상한이 낮으면 모델을 고쳐도 차이가 날 수 없다.

**설정**
- `src/iawm/oracle_planner.py`: 학습된 모델 대신 env의 실제 물리(agent 이동, 벽 clamp, obstacle push)를 300개 후보에 대해 batch로 계산하는 CEM planner다. CEM 설정(후보 300, 반복 30, top-k 30, 25 step 계획 후 25 step 실행)과 cost(블록 끝 target 거리의 합)는 모델 planning과 같다.
  - `oracle_aware`: obstacle 궤적과 push를 넣는다(완벽한 interaction 모델).
  - `oracle_blind`: obstacle을 뺀다. "obstacle은 agent에 영향을 주지 않는다"는 CF-JEPA 가정과 같은 모델이다.
- `python scripts/eval.py --config-name=collision_tworoom +world.obstacle_info=first policy=oracle_<mode> seed=<42|43|44> [조건]`. 평가 data와 시작 상태는 E26, E27과 같다.
- 검증: local CPU에서 simulator 예측 위치와 실제 env 위치를 1,000 step 비교했다. 평균 차이는 0.004px이다. Colab log의 sim error가 큰 것은 성공한 env를 World가 바로 reset해서 생긴 값이고, 이후 commit에서 reset을 감지하도록 고쳤다.
- Colab CPU 세션, 약 50분.

**결과** (success rate %, seed 3개 평균. 괄호는 seed별)

| 조건 | oracle_blind | oracle_aware | 차이 | 참고: 학습 모델 (E26, E27 P2) |
|---|---|---|---|---|
| b0 obstacle 없음 | | 98.0 (98 / 100 / 96) | | |
| c1b | 90.7 (90 / 92 / 90) | 98.0 (98 / 100 / 96) | +7.3 | 73~85 |
| **r3b** | **60.0** (60 / 62 / 58) | **94.7** (92 / 94 / 98) | **+34.7** | 49~54 |

**해석**
- **r3b에서 충돌을 정확히 예측하면 성공률이 35%p 오른다.** 이 조건은 interaction 모델링이 planning을 크게 바꿀 수 있는 조건이고, 상한이 낮아서 차이가 안 났던 것은 아니다.
- CF-JEPA 가정과 같은 blind simulator도 60%로 학습 모델(49~54%)보다 높다. 학습 모델과 정답 사이의 차이는 두 부분이다. 하나는 일반 예측 오차(blind 60 대 모델 50 내외)이고, 다른 하나는 interaction(aware 95 대 blind 60)이다. 지금 gate는 두 번째 부분을 전혀 얻지 못한다.
- 1-step 예측 개선(E27, contact 오차 −18.5%)이 이득으로 이어지지 않는 이유 후보: (a) planning은 5 block(25 step) 앞을 상상하는데, 상상한 z^u로는 gate가 미래 충돌을 맞히지 못한다. (b) push 결과를 정확히 맞히지 못해 충돌하는 후보의 cost가 충분히 커지지 않는다.

**다음 (E29):** (a)를 multistep 진단(상상 α의 AUROC)으로 확인한다. (b)를 우회하기 위해 gate α 자체를 충돌 회피 cost로 쓴다(cost += λ·d0·Σα). 같은 방식의 정답판(`oracle_avoid`, push는 모르고 충돌만 피함)으로 이 방법의 상한도 잰다.

**결과물:** `outputs/e28/res/e28/`(git 제외)에 평가 log와 `status.txt`가 있다.

## 2026-11-12 · E29. Gate α를 충돌 회피 cost로 쓰기, 상상 rollout에서의 α

**목적:** E28에서 r3b는 interaction을 정확히 예측하면 35%p 오를 수 있는 조건으로 확인됐다. push 결과를 정확히 맞히지 못해도 "충돌이 일어날지"만 알면 피할 수 있다. 그래서 gate α를 planning cost에 넣어 본다. 함께 상상 rollout 안에서 α가 미래 충돌을 맞히는지 확인한다.

**설정**
- `iawm.cfjepa_gate.CFJEPA.get_cost`: `contact_cost = λ > 0`이면 상상 rollout의 각 block에서 α(다음 block contact 확률로 학습됨)를 모아 cost += λ · d0 · Σ_k α_k를 더한다. d0는 현재 z^c와 goal z^c의 거리라서 λ는 단위가 없다. `scripts/eval.py`의 `+contact_cost=λ`로 켠다.
- 정답판 `oracle_avoid`: push는 모르는 blind simulator에 충돌 block마다 λ · d0를 더한다(λ = 1).
- 모델: `e27_gate`(epoch 10). 평가는 P2, seed 42/43/44.
- 진단: `multistep_error.py --ckpts e27_cf,e27_gate --data r3t_contact.h5 --steps 5 --distractors 0 --gate-modes learned,off`. r3t는 학습 data와 seed가 다른 obstacle 3개 data 600 episode다.
- L4, 약 1시간.

**결과: planning** (success rate %, seed 3개 평균)

| 조건 | gate (E27, cost 없음) | λ = 0.5 | λ = 1 | λ = 2 | oracle_blind | oracle_avoid | oracle_aware |
|---|---|---|---|---|---|---|---|
| r3b | 50.7 | 54.7 (56 / 52 / 56) | 47.3 (48 / 42 / 52) | 45.3 (50 / 46 / 40) | 60.0 | **71.3** (72 / 70 / 72) | 94.7 |
| c1b | 76.0 | | 67.3 (76 / 56 / 70) | | 90.7 | | 98.0 |

**결과: 상상 rollout** (k block 앞 z^c 오차 / persistence 오차. 낮을수록 좋다. contact는 1~k block 사이 충돌이 있는 clip)

| k | CF-JEPA contact | gate off contact | gate learned contact | gate learned non-contact | 상상 α AUROC |
|---|---|---|---|---|---|
| 1 | 0.312 | 0.310 | **0.259** | 0.050 | 0.964 |
| 2 | 0.362 | 0.362 | 0.317 | 0.051 | 0.903 |
| 3 | 0.415 | 0.407 | 0.380 | 0.061 | 0.869 |
| 5 | 0.503 | 0.486 | 0.471 | 0.084 | 0.816 |

**해석**
- **상상 rollout에서도 gate는 미래 충돌을 맞힌다.** 예측한 z^u만으로 5 block 앞 contact의 AUROC가 0.82다. "상상한 z^u로는 충돌을 모른다"는 후보 (a)는 아니다.
- **α를 cost로 써도 성공률은 오르지 않는다.** λ = 0.5의 54.7은 표준오차 안이다. λ를 키우면 오히려 떨어지고, c1b에서도 떨어진다. 정답판 avoid도 71%로 aware(95%)에 크게 못 미친다. 이 조건에서 obstacle 3개(반지름 22)는 피하기 어렵다. 이득의 대부분은 피하는 데서가 아니라, 밀린 뒤 어디에 있게 될지 아는 데서 나온다.
- **병목은 push 결과 예측이다.** contact clip의 z^c 오차는 non-contact의 5~6배이고, gate는 이를 k=1에서 17%, k=5에서 3%만 줄인다. h_int는 z^c, z^u, action만 보고 push를 바로 맞혀야 한다. 하지만 env는 "각자 움직인 뒤 겹치면 밀어낸다"로 동작하므로, push는 (agent가 갈 곳, obstacle이 갈 곳)의 함수로 보는 편이 단순하다.

**다음 (E30):** 학습된 모델을 고정하고 h_int 입력 후보를 새로 학습해 비교한다. 후보는 [z^c, z^u]와, 여기에 g의 예측(push 전 agent)과 h의 예측(다음 z^u)을 더한 입력(resolve)이다.

**결과물:** `outputs/e29/res/e29/`(git 제외)에 평가 log, `multistep.txt`, `status.txt`가 있다.

## 2026-11-13 · E30. 고정한 gate 모델에 h_int 입력 후보를 새로 학습 (resolve 입력 진단)

**목적:** E29에서 병목은 push 결과 예측이었다. env는 "각자 움직인 뒤 겹치면 밀어낸다"로 동작하므로, h_int 입력에 g의 예측(push 전 agent)과 h의 예측(다음 z^u)을 더하면(resolve) push를 맞히기 쉬워지는지 학습 전에 본다.

**설정**
- `fit_hint_frozen.py --ckpt e27_gate/weights_epoch_10.pt --data r3f_contact.h5 --train-episodes 3000 --test-episodes 500 --distractors 0 --lrs 1e-3 --steps 6000`. r3f는 obstacle 3개(반지름 22) 3,500 episode이고, 학습 data와 seed가 다르다(777).
- head: `tf_latent`, `mlp_latent`(기존 입력), `tf_resolve`, `mlp_resolve`(+ g 예측, h 예측), `mlp_resolve_true_u`(h 예측 대신 실제 다음 z^u). L4, 약 35분.

**결과** (test clip 6,025개, contact 47.8%. 값은 z^c 오차의 상대 감소. 높을수록 좋다)

| head | non-contact | contact | contact (pooled) |
|---|---|---|---|
| 모델 안의 h_int (함께 학습) | −1.9% | 7.7% | 12.0% |
| tf_latent | −7.5% | −32.2% | −15.7% |
| mlp_latent | −80.2% | −80.4% | −42.4% |
| tf_resolve | −1.2% | −30.3% | −14.4% |
| mlp_resolve | −74.7% | −71.6% | −38.7% |
| mlp_resolve_true_u | 49.8% | 54.0% | 62.2% |

**해석**
- 새로 붙인 head는 모두 test에서 오차를 늘렸다. 3,000 episode(3.6만 clip)에 6,000 step(약 85 epoch)을 학습해 과적합한 것으로 보인다. 같은 조건에서 resolve가 기존 입력보다 조금 덜 나쁘지만, 이 진단으로는 두 입력을 가를 수 없다.
- 실제 다음 z^u를 넣으면 non-contact에서도 50% 줄어든다. 다음 frame의 z^u에 agent 정보가 섞여 있다는 뜻이므로(leak), 이 값을 obstacle 예측이 완벽할 때의 상한으로 읽을 수는 없다.
- 결론: frozen fit은 판단 근거가 되지 못했다. resolve 입력은 `iawm.cfjepa_gate`에 `int_input='resolve'`로 넣었고, 실제 학습(E32)으로 비교한다.

**결과물:** `outputs/e30/res/e30/`(git 제외).

## 2026-11-13 · E31. 정답 simulator로 학습 조건 고르기

**목적:** E15에서 obstacle 3개 data는 정답 state로도 contact 잔차를 2~4%만 설명했다. r3b는 interaction 이득이 크지만(E28, +35%p) push가 복잡하다. obstacle이 적어 push가 단순하면서 interaction 이득이 큰 조건을 정답 simulator로 싸게 찾는다.

**설정:** `oracle_blind`, `oracle_aware`, seed 42 (50 episodes), budget 30. CPU 세션, 약 30분. 이 실행 뒤 simulator의 push 반복 종료 조건을 env와 같게(겹침 tol 1e-3) 고쳐 local에서 3,000 step 오차 0px을 확인했다. 이번 결과는 고치기 전 값이다(aware가 약간 낮게 나왔을 수 있다).

**결과** (success rate %, seed 1개)

| 조건 | blind | aware | 차이 |
|---|---|---|---|
| `r1b` 1개, 반지름 22 | 80 | 98 | +18 |
| **`r1x` 1개, 반지름 30** | **68** | **96** | **+28** |
| `f1b` 1개, 반지름 22, 속도 6~10 | 84 | 98 | +14 |
| `c2b` 2개, 반지름 14 | 82 | 98 | +16 |
| `r2b` 2개, 반지름 22 | 68 | 96 | +28 |
| `f2b` 2개, 반지름 22, 속도 6~10 | 78 | 98 | +20 |
| (E28) `r3b` 3개, 반지름 22 | 60 | 95 | +35 |

**결정:** obstacle 1개로 이득이 가장 큰 `r1x`를 다음 학습 조건으로 한다. obstacle 1개 data에서는 latent가 contact 잔차를 정답 state만큼 설명했다(E10).

**결과물:** Colab log는 받지 못했다(세션 종료 전 다운로드 실패). 위 표는 `outputs/e31/res/status_live.txt`의 값이다.

## 2026-11-13 · E32. r1x에서 CF-JEPA, gate, resolve gate 학습 (resolve는 OOM)

**목적:** E31에서 고른 `r1x`(obstacle 1개, 반지름 30) data로 CF-JEPA, 기존 gate(state 입력), resolve gate(E30)를 처음부터 학습해 planning을 비교한다.

**설정**
- data: `collect_collision.py --episodes 10000 --num-obstacles 1 --obstacle-radius 30 --name r1x` + contact 라벨. 727,377 frame, contact step 19.2%, contact 있는 episode 94.8%, goal 도달 59.5%.
- 공통: `data=collision_tworoom_contact data.dataset.name=r1x_contact.h5 distractor.num=0 trainer.max_epochs=10`.
- `e32_cf`: `model=cfjepa`. `e32_gate`: `model=cfjepa_gate model.predictor.gate_objective=supervised +model.int_lr=1e-3`. `e32_res`: gate + `+model.predictor.int_input=resolve`.
- A100 1장에서 3개를 병렬로 학습했다.

**결과:** `e32_res`가 28분 만에 CUDA OOM으로 죽었다(40GB에 3개 × 13GB). `e32_cf`, `e32_gate`는 10 epoch를 마쳤다. 평가 단계 전에 pipeline이 멈춰서 평가는 E33 세션에서 했다(아래 E33 표).

**교훈:** A100 40GB에서 이 모델의 full 학습은 2개까지만 병렬로 돌린다.

## 2026-11-13 · E33. Resolve gate 다시 학습, r1x planning 비교

**목적:** E32에서 죽은 resolve gate를 다시 학습하고, `e32_cf`, `e32_gate`와 같은 조건에서 비교한다.

**설정**
- `e33_res`: E32의 `e32_res`와 같다. `e33_resb`: resolve + `model.predictor.int_depth=4 model.int_aux_weight=3.0`. A100에서 2개 병렬, 10 epoch, 약 9.5시간.
- 평가: P2(`+obs_fix=true +plan_history=3`), `world.num_obstacles=1 eval.eval_budget=30 +world.obstacle_radius=30.0`, seed 42~49(seed당 50 episodes). oracle은 seed 42~44.
- 진단: `analyze_gate.py`(500 episodes), `multistep_error.py --steps 5 --gate-modes learned,off`.

**결과: planning** (success rate %, 평균 ± 표준오차, seed 8개. oracle은 3개)

| 모델 | gate learned | gate off |
|---|---|---|
| oracle (정답 simulator) | aware 97.3 ± 0.7 | blind 77.3 ± 3.7 |
| `e32_cf` CF-JEPA | 63.0 ± 2.0 | |
| `e32_gate` (state 입력) | 59.5 ± 2.3 | 62.2 ± 1.3 |
| `e33_res` (resolve 입력) | 59.5 ± 2.5 | 53.5 ± 2.8 |
| `e33_resb` (resolve, 큰 h_int) | 55.2 ± 2.9 | 55.3 (seed 3개) |

epoch 4 (seed 42~45): `e32_cf` 51.0, `e33_res` learned 54.0 / off 49.5, `e33_resb` 46.5 / 44.5.

**결과: 예측** (`multistep_error.py`, k = 1 / 5 block 앞 z^c 오차, persistence 대비)

| 모델 | non-contact k=1 | contact k=1 | contact k=5 | 상상 α AUROC k=1 / 5 |
|---|---|---|---|---|
| `e32_cf` | 0.027 | 0.107 | 0.233 | |
| `e32_gate` learned | 0.032 | 0.102 | 0.221 | 0.987 / 0.942 |
| `e33_res` learned | 0.040 | 0.109 | 0.225 | 0.990 / 0.938 |
| `e33_res` off | 0.041 | 0.200 | 0.288 | |

`analyze_gate.py`: `e33_res` α AUROC 0.979, contact z^c 오차 19.3% 감소(`e32_gate` 18.9%). `e33_resb` 26.3%.

**해석**
- **resolve gate에서 처음으로 gate를 켜는 것이 planning을 올렸다.** `e33_res`의 learned − off는 seed별로 +6, +4, +14, +2, +6, −6, +10, +12(평균 +6.0, 8개 중 7개 양수)다. 기존 gate는 켜면 2.7%p 내려갔다.
- **하지만 모델 전체는 CF-JEPA보다 3.5%p 낮다.** 같이 학습하면서 gate를 끈 부분(encoder, g)이 나빠졌다. non-contact 오차가 0.027 → 0.040으로 늘었고, gate off 성공률이 53.5로 CF-JEPA보다 9.5%p 낮다. resolve 입력의 h_int가 contact 오차를 맡으면서 g가 contact를 덜 배운 것으로 보인다(off의 contact 오차 0.200 대 CF-JEPA 0.107).
- 큰 h_int(`resb`)는 contact 오차를 더 줄였지만(26%) planning은 더 낮다.
- oracle blind seed 42가 E31의 68과 달리 82다. E31은 log가 없어 원인을 확인하지 못했다.

**다음 (E35):** 학습된 CF-JEPA를 고정하고 interaction module만 학습한다. gate off가 CF-JEPA와 같아지므로 gate on의 이득이 그대로 CF-JEPA 대비 이득이 된다.

**결과물:** `outputs/e33/res/e33/`(git 제외), Proton `results/E33_r1x_resolve/`.

## 2026-11-13 · E34. 학습 없이 planning cost 바꾸기 (e32_gate, r1x)

**목적:** `e32_gate`는 one-step 예측이 좋은데(contact 오차 19% 감소, α AUROC 0.978) 켜면 planning이 내려간다. CEM이 상상한 push가 유리하게 나온 plan을 고르는 것(model exploitation)인지 본다.

**설정:** E33 세션에서 학습과 같이 돌렸다. P2, r1x, seed 42~49.
- `+plan_combine=max|mean` (`iawm.cfjepa_gate.CFJEPA.get_cost`): gate on rollout과 off rollout의 cost를 합친다. max는 "interaction이 있어도 없어도 좋은 plan"을 고른다.
- `+contact_cost=λ` (E29): 상상 α로 충돌 회피 cost를 더한다. 정답판 `oracle_avoid`(λ = 1, seed 42~44)도 쟀다.

**결과** (success rate %, seed 8개 평균)

| 방법 | 성공률 |
|---|---|
| gate learned / off (E33) | 59.5 / 62.2 |
| `plan_combine=max` | 58.2 ± 2.7 |
| `plan_combine=mean` (seed 3개) | 60.0 |
| `contact_cost=0.25` | 59.5 ± 3.0 |
| `contact_cost=0.5` | 54.8 ± 2.3 |
| oracle blind / avoid / aware (seed 3개) | 77.3 / 82.0 / 97.3 |

**해석:** 어느 방법도 gate off나 CF-JEPA를 넘지 못했다. 정답판에서도 회피는 +4.7%p뿐이고 push 결과를 아는 aware가 +20%p다. r1x에서도 이득의 대부분은 push 결과 예측에서 나온다.

## 2026-11-14 · E35. 학습된 CF-JEPA를 고정하고 interaction module만 학습 (r1x)

**목적:** E33에서 같이 학습하면 base(encoder, g, h)가 나빠졌다. `e32_cf`를 고정하고 interaction module(`h_int`, `int_in`, `int_out`, gate)만 학습하면 gate off가 CF-JEPA와 같아지므로, gate on의 이득이 곧 CF-JEPA 대비 이득이 된다.

**설정**
- `scripts/train.py`의 `+init_from=e32_cf/weights_epoch_10.pt +freeze_base=true`(학습 parameter 0.808M). 6 epoch, L4.
- `e35_fres`: resolve 입력, `e35_fstate`: state 입력. 둘 다 `gate_objective=supervised`, `int_lr=1e-3`.
- 평가: P2, r1x(`world.num_obstacles=1 eval.eval_budget=30 +world.obstacle_radius=30.0`). CF-JEPA도 같은 L4에서 다시 쟀다(`cfl4`). CEM 결과는 같은 GPU에서도 실행마다 seed당 ±4~8%p 흔들리므로 seed를 16개(42~57)로 늘렸다(seed 50~57은 E36 세션에서 쟀다).
- log는 내려받기 실패로 잃었고 status 파일만 남았다(E36부터 재시도와 큰 파일 제외로 고쳤다).

**결과** (success rate %, 평균 ± 표준오차)

| 모델 | seed 42~49 | seed 50~57 | 16개 |
|---|---|---|---|
| `e32_cf` CF-JEPA (L4) | 61.5 ± 1.6 | 57.5 ± 2.6 | 59.5 ± 1.6 |
| `e35_fres` epoch 6 | 63.2 | 60.5 | 61.9 ± 1.6 |
| `e35_fres` epoch 3 | 62.8 | | |
| `e35_fstate` epoch 6 | 60.8 ± 1.4 | | |

- `e35_fres` − CF-JEPA (seed별 paired): 평균 +2.4 ± 1.5%p, 16개 중 9개 높고 5개 낮다. 유의하지 않다.
- 예측(`multistep_error.py`, E36 세션): `e35_fres`의 contact z^c 오차는 k=1에서 0.100(off 0.107, CF-JEPA 0.107), k=5에서 0.229(0.233). non-contact는 CF-JEPA와 같다(0.027). 상상 α AUROC k=1 / 5는 0.991 / 0.936이다.

**해석:** base를 고정하니 E33의 base 악화는 사라졌지만, resolve h_int가 고친 양이 작다(contact k=1 오차 6.5% 감소). planning 차이도 그만큼 작아서 noise 안에 있다.

## 2026-11-15 · E36. CF-JEPA에서 시작해 gate 모델 전체를 낮은 lr로 fine-tune (r1x)

**목적:** E35는 base를 완전히 고정해서 h_int가 고칠 수 있는 폭이 작았을 수 있다. base를 `e32_cf`에서 시작해 낮은 lr(1e-5)로 같이 움직이고, interaction module은 1e-3으로 학습한다.

**설정:** `+init_from=e32_cf/weights_epoch_10.pt optimizer.lr=1e-5`, resolve, supervised, `int_lr=1e-3`, 3 epoch, L4 1개(`outputs/e36/e36_pipeline.sh`). epoch 1 check(seed 42~45)에서 이득이 보이지 않아 마지막 평가를 gate learned, seed 50~57로 줄였다(`outputs/e36/trim.sh`). 세션 약 4.5시간, 약 9 CU.

**결과**

| 모델 | 성공률 |
|---|---|
| CF-JEPA (L4), seed 42~45 / 50~57 | 59.0 / 57.5 ± 2.6 |
| `e36_ft` epoch 1 learned / off, seed 42~45 | 59.0 / 61.0 |
| `e36_ft` epoch 3 learned, seed 50~57 | 61.8 ± 2.5 |

- epoch 3 − CF-JEPA (paired, seed 50~57): +4.3 ± 2.1%p, 8개 중 6개 높고 1개 낮다.
- **하지만 h_int는 아무것도 고치지 않는다.** `analyze_gate.py`: α AUROC 0.978(gate는 contact를 잘 찾는다), 그러나 α = 1로 켰을 때 contact z^c 오차 변화가 0.05%(0.10673 → 0.10669)다. `multistep_error.py`에서도 learned와 off가 k=1~5 모두 같다(contact k=1 0.103, k=5 0.239).

**해석**
- 예측이 gate on과 off에서 같으므로, epoch 3의 +4.3%p는 gate가 아니라 base를 r1x data로 3 epoch 더 학습한 효과이거나 평가 noise다. 제안 module의 효과로 주장할 수 없다. 이를 확인하려면 gate 없는 CF-JEPA를 같은 조건으로 3 epoch 더 학습한 대조군이 필요하다.
- base가 움직일 수 있으면 forward loss가 base 쪽에서 먼저 줄어들고, h_int의 잔차는 0 근처에 머문다. E33(처음부터 같이 학습)과 반대로, 이번에는 base가 contact를 맡았다.
- 지금까지 gate가 CF-JEPA 대비 이득을 낸 설정은 base 고정(E35)뿐이고, 그 크기(+2.4%p)는 noise 수준이다.

**결과물:** `outputs/e35/res/`, `outputs/e36/res/`(git 제외).

## 2026-11-15 · E37. Gate α로 다시 계획할 시점 정하기 (gate-triggered replanning, 학습 없음)

**목적:** E25~E36에서 gate는 충돌을 잘 찾지만(AUROC 0.98) push 결과는 잘 맞히지 못했다. 지금 평가는 한 번 세운 계획을 25 step(5 block) 그대로 실행해서 push 예측 오차가 그대로 실패로 이어진다. gate α가 충돌을 예상한 block까지만 실행하고 실제 관측으로 다시 계획하면, gate가 잘하는 것(감지)만 쓰고 못하는 것(push 예측)은 관측으로 대신할 수 있는지 본다.

**구현** (`scripts/eval.py`, `src/iawm/oracle_planner.py`)
- `+replan=contact` (정답 simulator): blind 계획 경로가 obstacle과 처음 겹치는 block까지만 실행한다.
- `+replan=gate +replan_tau=τ`: CEM이 고른 계획을 한 번 더 상상 rollout해서 block마다 α를 읽고, 처음 τ를 넘는 block까지만 실행한다.
- `+replan=random +replan_p=p`: 대조군. block마다 확률 p로 끊는다.
- `plan_config.receding_horizon=k`: k block마다 다시 계획(고정 간격 대조군).
- 모든 평가에서 계획 횟수(`[plans]`)를 기록한다.

**설정:** r1x, P2, L4 1개. 정답 simulator는 seed 42~44, 학습 모델은 seed 42~49. gate 모델은 `e35_fres`(base가 `e32_cf`와 같다), 대조군은 `e32_cf`.

**결과** (success rate %, 평균 ± 표준오차. 계획 횟수는 episode당)

| 방법 | seed | 성공률 | 계획 횟수 |
|---|---|---|---|
| 정답 blind, 25 step 고정 | 3 | 70.0 ± 4.6 | 2.00 |
| 정답 blind, 충돌 직전에만 다시 계획 | 3 | 92.7 ± 1.3 | 4.05 |
| 정답 blind, 5 step마다 | 3 | 95.3 ± 2.9 | 6.00 |
| 정답 aware, 25 step 고정 | 3 | 96.7 ± 1.3 | 2.00 |
| CF-JEPA, 25 step 고정 (E35, 16개) | 16 | 59.5 ± 1.6 | 2 |
| CF-JEPA, 10 step마다 | 4 | 74.5 ± 3.2 | 2.06 |
| CF-JEPA, 5 step마다 | 8 | 78.8 ± 1.5 | 3.35 |
| CF-JEPA, 무작위 p = 0.3 | 4 | 67.0 ± 1.9 | 1.96 |
| CF-JEPA, 무작위 p = 0.6 | 5 | **79.2 ± 2.3** | 2.45 |
| gate τ = 0.5 | 8 | 77.5 ± 1.2 | 2.89 |
| gate τ = 0.9 | 5 | 74.0 ± 1.7 | 2.62 |

- gate τ = 0.9 − 무작위 p = 0.6 (paired, seed 42~46): −12, −2, −4, −8, 0. gate가 계획을 더 하고도 낮다.
- gate는 계획의 약 75%(τ = 0.5) / 65%(τ = 0.9)를 첫 block에서 끊었다. 나머지는 대부분 5 block을 모두 실행했다.

**해석**
- **다시 계획하기만 해도 성공률이 60에서 75~79로 오른다.** 정답 simulator에서도 다시 계획하면 blind가 aware에 거의 닿는다(95.3 대 96.7). 25 step 고정 실행 평가는 사실상 open-loop라 interaction 예측의 비중을 크게 만든다.
- **gate로 시점을 고르는 것은 무작위보다 낫지 않다.** 정답 simulator에서는 충돌 직전 재계획이 통했지만(92.7), 학습 모델은 충돌 밖의 일반 예측 오차도 커서 충돌만 골라 고치는 것으로는 부족하다. gate는 obstacle이 가까우면 바로 끊고, 충돌이 없다고 보면 25 step을 그대로 실행해 일반 오차를 고칠 기회를 놓친다.
- 계획 횟수가 같으면 성공률은 시점보다 횟수와 간격의 고름에 좌우된다.
- seed 46 이후 phase B는 결론이 분명해서 중단했다.

**사고:** 수동 중단에 쓴 `pkill -f`가 `colab exec` 자신의 shell까지 죽여 이후 exec가 멈췄다. watcher가 완료를 보지 못해 세션이 약 3.5시간 빈 채로 돌았고(약 7 CU) log 다운로드도 실패했다. 위 표는 `outputs/e37/res/status_live.txt`와 직접 확인한 status 값이다. E37 전체 약 12 CU.

**결과물:** `outputs/e37/res/status_live.txt`(git 제외).

## 2026-11-16 · E38. 진단: CF-JEPA latent의 obstacle 정보와 contact 잔차 (학습 없음)

**목적:** E25~E37에서 gate는 contact를 잘 찾지만(AUROC 0.98) CF-JEPA 대비 contact z^c 오차를 최대 6.5%만 줄였다. 원인 가설은 두 가지다.
1. CF-JEPA는 g가 z^u를 보지 못할 뿐, encoder가 obstacle 정보를 z^c에 넣는 것은 막지 않는다. 그래서 Collision data로 학습한 CF-JEPA는 이미 obstacle을 (흐릿하게) 쓰고 있고, gate가 더하는 정보가 적다. 근거는 간접적이다(E8 probe의 z^c → obstacle x R² 0.85, E10 `n0`의 zc 0.410 대 zc+zu 0.445, E33에서 h_int가 contact를 맡았을 때 g의 contact 오차 0.200 대 CF-JEPA 0.107). r1x 모델(`e32_cf`)에서 직접 잰 적은 없다.
2. 모든 모델의 contact 오차가 0.10 근처에서 멈춘 것은 latent의 obstacle 상태가 정밀하지 않기 때문이다. 예전 모델에서는 정답 state를 주면 h_int가 contact 오차를 26~44% 줄였다(E19, E20). 반대로 distractor 없는 모델에서는 정답 state도 latent보다 더 설명하지 못한 적이 있다(E10 `n0`, E15).

**설정**
- 대상: `e32_cf`(epoch 10, r1x로 학습한 CF-JEPA). 고정한다.
- data: `r1y`. r1x와 같은 조건(obstacle 1개, 반지름 30)이고 수집 seed만 다르다(4242, 1만 episode). `e32_cf`가 학습에 쓰지 않은 data에서 잰다. E9~E35의 분석은 학습에 쓴 data(같은 seed로 다시 수집)에서 쟀다.
- `probe_latent.py --ckpt e32_cf/weights_epoch_10.pt --data r1y.h5 --episodes 2000 --distractors 0`: frame 한 장의 z^c, z^u에서 obstacle과 agent 위치를 읽는다. 이번에 px 단위 오차와 contact frame별 값을 더했다. obstacle은 agent 위에 그려지므로(`envs/collision_tworoom.py`의 `_render_frame`), contact frame에서 agent 위치가 흐려지는지도 본다.
- `probe_residual.py --ckpt e32_cf/weights_epoch_10.pt --data r1y.h5 --episodes 10000`: contact 잔차를 입력별로 얼마나 설명하는지(B) 잰다. 입력은 `zc`, `zc+zu`, `zc+zu+obs`(latent + 정답 obstacle 위치·속도·5 step 뒤 위치, E39의 h_int가 보는 정보), `oracle`(정답 state만)이다. 이번에 history 3 frame의 latent에서 obstacle 위치와 속도를 읽는 state probe도 더했다.

**읽는 법 (실행 전에 정함)**
- 가설 1: z^c만으로 obstacle 위치가 z^u와 비슷한 정밀도로 읽히면 맞다. z^c에서 거의 읽히지 않으면(MLP 오차가 obstacle 위치의 표준편차에 가까우면) 틀렸고, 진단을 다시 세운다.
- 가설 2: contact B가 `zc+zu+obs` ≥ 0.30이고 `zc+zu`보다 0.15 이상 높으면 맞다(정밀한 obstacle 상태가 병목). `oracle`과 `zc+zu+obs`가 모두 낮으면(0.15 미만) 정답 state로도 못 맞히는 것이므로, 예측 개선 방향을 접는 근거가 된다.
- E38은 E39와 같은 세션에서 돌린다. E39의 학습이 싸져서(아래) E38 결과로 E39 학습을 막지 않는다. 비용이 큰 planning 평가만 E39의 held-out 수치로 판정한다.

**결과: frame 한 장의 latent에서 읽은 위치** (`probe_latent.py`, r1y 2,000 episode, test 28,657 frame. 각 칸은 linear | MLP)

| | z^c | z^u | z 전체 |
|---|---|---|---|
| obstacle x, y (R²) | 0.989, 0.996 \| 0.997, 0.999 | 0.376, 0.466 \| 0.948, 0.957 | 0.990, 0.997 \| 0.998, 0.999 |
| agent x, y (R²) | 0.989, 0.999 \| 0.998, 0.999 | 0.501, 0.600 \| 0.955, 0.958 | 0.991, 0.999 \| 0.999, 0.999 |
| obstacle 위치 오차, contact frame (px) | 4.50 \| 1.79 | 46.09 \| 11.52 | 4.49 \| 1.82 |
| agent 위치 오차, non-contact frame (px) | 3.72 \| 1.56 | 38.65 \| 11.24 | 3.34 \| 1.66 |
| agent 위치 오차, contact frame (px) | 3.90 \| 1.74 | 49.50 \| 12.78 | 3.67 \| 1.66 |

**결과: contact 잔차 설명력과 state probe** (`probe_residual.py`, r1y 1만 episode, test transition 22,999개, contact 31.9%)

| 입력 | non-contact B | contact B |
|---|---|---|
| `zc` | 0.105 | 0.029 |
| `zc+zu` | 0.110 | 0.031 |
| `zc+zu+obs` | 0.156 | 0.141 |
| `oracle` | 0.156 | 0.115 |

| history 3 frame의 latent → 마지막 frame의 state (non-contact / contact, 2D 오차 평균) | z^c | z^u | z^c + z^u | 값의 표준편차 |
|---|---|---|---|---|
| agent 위치 (px) | 0.87 / 1.15 | 7.11 / 7.70 | 0.97 / 1.15 | 68.35 |
| obstacle 위치 (px) | 2.32 / 1.69 | 7.47 / 7.40 | 2.05 / 1.68 | 65.30 |
| obstacle 속도 (px/step) | 1.83 / 1.66 | 2.15 / 2.12 | 1.78 / 1.63 | 4.53 |

**해석**
- **가설 1은 맞다. z^c와 z^u의 분리가 무너져 있다.** `e32_cf`의 z^c에서 obstacle 위치가 1.8px(MLP)로 읽힌다. z^u(11.5px)보다 훨씬 정밀하다. agent 위치도 z^c에 있고 z^u에는 흐릿하게만 있다. Collision data로 학습한 CF-JEPA는 obstacle을 z^c에 넣어 g가 직접 쓰고, z^u는 거의 쓰이지 않는 사본이다. gate가 z^u에서 가져올 새 정보가 없었던 이유다.
- **가설 2는 미리 정한 기준으로는 확인되지 않았다.** contact B는 `zc+zu+obs` 0.141, `oracle` 0.115로 둘 다 0.15 미만이다. 미리 정한 읽는 법대로면 "정답 state로도 못 맞힌다"에 해당한다.
- **그러나 이 probe가 약한 도구였다.** 같은 정보(latent + 정답 obstacle)로 실제 h_int를 학습한 E39에서는 contact 오차가 test data에서 67~74% 줄었다. MLP probe(4,000 step, 마지막 position만)는 그 일부만 잡는다. E9, E10, E15에서 이 probe의 B로 내린 "정답 state로도 못 맞힌다"류의 판단은 같은 이유로 과소평가였을 수 있다. 앞으로 이 질문은 probe가 아니라 `train_int_frozen.py`의 held-out 수치로 잰다.
- **부족한 것은 obstacle의 속도다.** 위치는 3 frame에서 1.7~2.3px로 읽히지만 속도 오차는 1.6~1.8px/step이다(속도의 표준편차 4.5). 모델의 한 step은 env 5 step이므로, 한 step 뒤 obstacle 위치가 8~9px 어긋난다. E39의 px 진단에서 CF-JEPA가 예측한 z^c의 obstacle 위치는 1 step 앞에서 이미 6.7~9.1px 틀렸다. 속도만 따로 넣어 확인한 것은 아니다.
- obstacle이 agent를 가려 위치가 흐려지는 문제는 없다(contact frame의 agent 위치 1.7px).

**결과물:** `outputs/e38/res/e38/`(git 제외)의 `e38_plat.log`, `e38_pres.log`.

## 2026-11-16 · E39. 상한: 정답 obstacle 궤적을 h_int에 넣은 gate (base 고정)

**목적:** "z^u가 obstacle을 완벽하게 담고 완벽하게 예측한다면, 지금의 gate 구조(z^c 예측에 α·h_int를 더한다)가 planning 성공률을 얼마나 올리는가"를 잰다. obstacle은 action과 agent의 영향을 받지 않고 움직이므로, planning 중에도 env 상태에서 미래 obstacle 궤적을 정확히 계산해 넣을 수 있다. 실제로 쓸 모델이 아니라 상한이다. 이득이 크면 obstacle 상태를 화면에서 읽고 예측하는 module(E40)을 만들 근거가 되고, 이득이 없으면 이 구조로는 정보가 완벽해도 이길 수 없다는 뜻이다.

**구현**
- `src/iawm/obs_oracle.py`: frame마다 [obstacle 위치, 속도, 5 step 뒤 위치](obstacle당 6차원, 고정 scale). 학습 data에서는 `obstacle_pos`, `obstacle_vel` column에서, planning에서는 env에서 계산한다(시작 전 frame은 `render_past`와 같은 역방향 계산).
- `InteractionGatedPredictor`의 `int_oracle_key='obs_oracle'`, `iawm.cfjepa_gate.CFJEPA.rollout`: rollout의 예측마다 그 window의 frame에 해당하는 값을 h_int에 넣는다. gate 입력에는 넣지 않는다.
- `scripts/eval.py`: 모델이 `obs_oracle` 입력이면 `HistoryPolicy`가 history frame의 obstacle 상태를 기록하고 미래 상태를 계산해 info의 `obs_oracle`로 넘긴다.
- `scripts/train_int_frozen.py`: base를 고정하고 interaction module만 학습한다. loss는 E35(`train.py +freeze_base=true`)와 같은 식이다. encoder가 고정이므로 frame latent를 한 번만 계산해 두고 그 위에서 학습한다(E35는 epoch당 약 55분).
- `scripts/colab/check_obs_oracle.py`: env를 돌리며 과거·미래 obstacle 상태 계산이 실제 궤적과 맞는지 확인한다(smoke test에서 실행).

**E35와 달라진 점:** E35는 `e32_cf`의 학습 data(r1x, 같은 seed)로 h_int를 학습했다. 그러면 h_int가 배우는 잔차는 base가 이미 본 data에서의 잔차다. 이번에는 seed가 다른 `r1y`로 학습한다.

**설정**
- base: `e32_cf/weights_epoch_10.pt`(고정). 학습 data `r1y`(마지막 500 episode는 검증용), test data `r1yt`(seed 777, 1,500 episode, base와 h_int 모두 보지 않은 data).
- 모델 (모두 resolve 입력, supervised gate, lr 1e-3, 30 epoch, 검증 오차가 가장 낮은 epoch를 쓴다)
  - `e39_obs`: 정답 obstacle 입력, h_int 크기는 E35와 같다(depth 2, hidden 128).
  - `e39_obsbig`: 정답 obstacle 입력, 큰 h_int(depth 4, hidden 256, mlp 1024).
  - `e39_lat`: 정답 없음(대조군). E35 `e35_fres`와 같은 구조를 새 data로 학습한 것이다. 정답 obstacle의 효과와 "새 data로 다시 학습한 효과"를 가른다.
- 예측 판정 (test data, contact transition의 z^c 오차, 학습된 α로 더한 예측): CF-JEPA 대비 감소율이 `e39_obs`, `e39_obsbig` 중 높은 쪽과 `e39_lat` 모두 15% 미만이면 planning 평가를 하지 않고 끝낸다. `analyze_gate.py`, `multistep_error.py`로 같은 수치를 따로 확인한다.
- planning 평가: P2, r1x(`world.num_obstacles=1 eval.eval_budget=30 +world.obstacle_radius=30.0 +obs_fix=true +plan_history=3`), L4 한 세션에서 seed를 짝지어 잰다. 대상은 `e32_cf`(CF-JEPA), 정답 obstacle 모델 중 test 수치가 높은 것, `e39_lat`이다.
  - 1단계: seed 42~49. 정답 obstacle 모델 − CF-JEPA의 paired 평균이 +2%p 미만이면 여기서 끝낸다.
  - 2단계: seed 50~57을 더해 16개로 판정한다.
- 성공 기준: 정답 obstacle 모델 − CF-JEPA (paired, seed 16개)가 +5%p 이상이고 표준오차의 3배 이상. `e39_lat` − CF-JEPA도 함께 보고한다.
- 비용 상한: 세션 1개(L4), 20 CU. 기한은 예상 시간 + 1시간으로 `colab-guard`에 등록한다.

**추가 진단 (11/16 23:37, 1단계 결과를 본 뒤 같은 세션에 더했다. 미리 정한 판정 대상이 아니라 탐색용이다)**
- 배경: test data에서 `e39_obsbig`의 contact z^c 오차는 73.7% 줄었고 5 block 상상 rollout에서도 contact 오차가 0.274에서 0.086으로 줄었다. 그런데 planning 1단계(seed 42~49)는 CF-JEPA 대비 +2.5 ± 2.1%p였다. 또 h_int를 항상 켜면(α = 1) non-contact 오차도 75% 줄었다. z^c에 obstacle 위치가 들어 있어서(E38), z^c 오차의 큰 부분이 agent가 아니라 obstacle의 다음 위치를 맞히는 오차일 수 있다.
- `scripts/multistep_px.py`: 실제 frame의 z^c에서 agent와 obstacle 위치(px)를 읽는 probe를 학습해 상상 rollout의 z^c에 적용한다. k block 앞 agent 위치 오차(px)를 CF-JEPA, `e39_obsbig`(learned, always), `e39_lat`에서 잰다. agent 위치 오차가 크게 줄었는데도 planning이 그대로면 병목은 planning 쪽(cost, CEM)이고, 거의 줄지 않았으면 z^c 오차 감소가 주로 obstacle 성분이었다는 뜻이다.
- `+gate_mode=always`로 `e39_obsbig`를 seed 42~49에서 평가한다(`r1x_obsalways`). gate가 닫혀 쓰이지 않던 non-contact 구간의 개선이 planning에 영향을 주는지 본다. 같은 세션의 CF-JEPA와 seed를 짝지어 비교한다.

**결과: 예측** (test data `r1yt`, 16,865 clip. z^c 오차, off는 CF-JEPA와 같다)

| 모델 (가장 좋은 epoch) | contact off → learned (감소율) | contact α = 1 감소율 | non-contact learned 감소율 | α AUROC |
|---|---|---|---|---|
| `e39_obsbig` (25) | 0.1129 → 0.0297 (**73.7%**) | 82.0% | 24.9% | 0.975 |
| `e39_obs` (28) | 0.1129 → 0.0376 (66.7%) | 74.9% | 22.3% | 0.975 |
| `e39_lat` (5) | 0.1129 → 0.1067 (5.5%) | 5.2% | 2.5% | 0.974 |

- 예측 판정(15%)은 정답 obstacle 모델이 통과했다. planning에는 `e39_obsbig`를 썼다.
- `analyze_gate.py`로 따로 잰 `e39_obsbig`의 값도 같다(contact err_off 0.1129, α = 1에서 0.0203). 새 학습 script의 수치가 기존 분석 script와 맞는다.
- 학습은 frame latent 계산 20분(73만 frame) 뒤 epoch당 약 80초(세 모델 동시)였다. E35는 epoch당 약 55분이었다.

**결과: 상상 rollout** (`multistep_error.py`, r1yt 11,601 clip. persistence 대비 z^c 오차. k = 1 / 3 / 5 block)

| 모델 | non-contact | contact | 상상 α AUROC |
|---|---|---|---|
| `e32_cf` | 0.031 / 0.031 / 0.048 | 0.123 / 0.212 / 0.274 | |
| `e39_obsbig` learned | 0.026 / 0.027 / 0.043 | 0.038 / 0.056 / **0.086** | 0.987 / 0.975 / 0.962 |
| `e39_lat` learned | 0.030 / 0.031 / 0.048 | 0.110 / 0.203 / 0.268 | 0.987 / 0.952 / 0.918 |

**결과: 상상 rollout을 px로 읽은 것** (`multistep_px.py`, 추가 진단. 실제 frame의 z^c에서 위치를 읽는 probe의 오차는 agent 0.70px, obstacle 1.29px. k = 1 / 3 / 5 block)

| 모델 | agent 위치 오차 non-contact (px) | agent 위치 오차 contact (px) | 예측한 z^c의 obstacle 위치 오차 contact (px) |
|---|---|---|---|
| `e32_cf` | 2.1 / 3.5 / 4.8 | 4.9 / 11.1 / **15.9** | 6.7 / 11.0 / 14.8 |
| `e39_lat` learned | 2.1 / 3.5 / 4.7 | 4.5 / 10.6 / 15.6 | 6.5 / 10.8 / 14.2 |
| `e39_obsbig` learned | 1.9 / 3.3 / 4.5 | 2.0 / 4.2 / **6.5** | 2.1 / 5.2 / 7.4 |
| `e39_obsbig` always (α = 1) | 1.2 / 1.7 / 2.4 | 1.9 / 3.0 / 4.2 | 1.7 / 2.1 / 2.5 |

**결과: planning** (r1x, P2, L4 한 세션, seed를 짝지음. success rate %, 평균 ± 표준오차)

| 모델 | seed 42~49 | seed 50~57 | 16개 |
|---|---|---|---|
| `e32_cf` CF-JEPA | 64.2 | 58.8 | 61.5 ± 1.6 |
| `e39_obsbig` (정답 obstacle) | 66.8 | 63.0 | 64.9 ± 2.1 |
| `e39_lat` (대조군) | 61.8 | 58.5 | 60.1 ± 1.9 |
| `e39_obsbig` `+gate_mode=always` (추가) | 66.8 ± 2.6 | | |

| paired 차이 (seed 16개) | 평균 ± 표준오차 | 높음 / 낮음 |
|---|---|---|
| 정답 obstacle − CF-JEPA | **+3.4 ± 1.7** | 11 / 4 |
| 대조군 − CF-JEPA | −1.4 ± 1.8 | 5 / 10 |
| 정답 obstacle − 대조군 | +4.8 ± 1.5 | 12 / 2 |
| always − CF-JEPA (seed 42~49) | +2.5 ± 2.0 | 5 / 2 |

- seed별 값: CF-JEPA 66 64 64 58 68 58 66 70 64 66 52 68 62 56 48 54 / 정답 obstacle 76 64 70 62 60 66 64 72 76 74 44 70 64 60 64 52 / 대조군 60 68 60 56 58 66 60 66 64 62 40 64 64 68 60 46 / always 64 68 74 62 76 58 58 74.
- 1단계의 정답 obstacle − CF-JEPA가 +2.5로 +2 기준을 넘어 2단계를 진행했다.

**해석**
- **미리 정한 성공 기준(+5%p 이상, 표준오차 3배 이상)에 미치지 못했다.** 정답 obstacle − CF-JEPA는 +3.4 ± 1.7%p(표준오차의 2.0배)다. 방향은 일관되지만(16개 중 11개 높음) 상한 실험으로서는 작다. 정답 simulator에서 interaction을 아는 것의 이득은 +20%p였다(E33).
- **정답 obstacle 상태를 주면 충돌 결과 예측은 크게 좋아진다.** contact z^c 오차가 74% 줄고, 5 block 앞 agent 위치 오차가 contact clip에서 15.9px에서 6.5px로 준다(성공 기준은 목표 16px 이내). E8~E36에서 "h_int가 push를 6~19%만 고친다"던 병목은 구조나 학습 방식이 아니라 obstacle 정보의 정밀도였다.
- **대조군은 달라지지 않았다.** base가 보지 않은 data로 학습해도 latent만 받은 h_int는 5.5%다(E35는 6.5%). E35의 약한 이득은 학습 data가 in-sample이어서가 아니었다. planning에서도 대조군은 CF-JEPA와 같거나 낮다(−1.4 ± 1.8).
- **예측 개선이 planning 성공으로 이어지지 않는다.** agent 위치 예측이 2.4배 정확해졌는데 성공률은 +3.4%p다. α = 1로 non-contact 예측까지 고쳐도(agent 위치 오차 2.4px / 4.2px) 성공률은 learned와 같다(66.8). 병목이 push 예측에서 planning 쪽으로 옮겨 갔다. 다만 예측 오차는 수집 data의 action(scripted policy)에서 잰 것이다. CEM이 고르는 계획에서의 예측 오차는 재지 않았다(2607.10362가 지적한 차이). 후보 원인은 다음과 같고, 이번 실험으로는 가르지 못한다.
  1. planning cost가 z^c 거리인데 z^c에 obstacle 위치가 들어 있다(E38). goal image에는 obstacle이 없으므로, cost에 agent와 무관한 항이 섞인다.
  2. CEM이 data에 없는 action 열에서 모델이 틀리는 계획을 고른다.
  3. 정답 blind simulator(77)와 CF-JEPA(60 안팎)의 차이처럼, interaction과 무관한 일반 오차가 여전히 성공률을 누른다.
- 한계: 이 모델은 정답 obstacle 궤적을 받으므로 실제로 쓸 수 있는 모델이 아니다. 상한으로만 읽는다. `e39_obsbig`는 `e39_obs`와의 비교(test 수치)로 골랐고 planning은 한 모델만 쟀다.

**다음:** E40(obstacle 상태를 화면에서 읽는 module)은 이 상한이 +3.4%p이므로 그대로 진행할 근거가 약하다. 먼저 "예측이 정확해져도 planning이 따라오지 않는 이유"를 가른다. 후보는 CEM이 고른 계획에서의 px 예측 오차 측정, agent 위치만 남긴 planning cost다. 팀과 정한 뒤 번호를 붙인다.

**비용과 운영:** 세션 `e38` 4시간 34분(20:13~00:47), 8.85 CU(285.67 → 276.82). smoke test가 첫 실행에서 검사 script의 예외 상황(episode 첫 frame에서 obstacle이 방 경계에 붙어 시작)을 잡았고, 수집을 멈추지 않고 고쳐 다시 올렸다. 중간에 로컬 PC를 재부팅했고 감시 서비스가 다시 떠서 결과 저장과 세션 종료까지 마쳤다.

**결과물:** `outputs/e38/res/`(git 제외)에 log 묶음 `e38_logs.tgz`, `status_live.txt`, `stable-wm/checkpoints/e39_*/`의 `config.json`, `metrics.json`, `int_only_best.pt`(학습한 interaction module만. `e32_cf`와 합치면 복원된다).

## 2026-11-17 · E40. Planning 진단: cost와 planner가 고른 계획에서의 예측 (학습 없음)

**목적:** E39에서 정답 obstacle 입력 모델은 5 block 앞 agent 위치 오차를 contact clip에서 15.9px에서 6.5px로 줄였지만 planning 성공률은 +3.4 ± 1.7%p였다. 예측이 정확해져도 planning이 따라오지 않는 이유를 두 가지로 나눠 잰다.
1. **cost:** planning cost는 z^c 거리인데 z^c에 obstacle 위치가 들어 있다(E38). goal image에는 obstacle이 없으므로 cost에 agent와 무관한 항이 섞인다.
2. **planner가 고른 계획에서의 예측:** E39의 예측 오차는 수집 data의 action에서 잰 것이다. CEM은 모델이 낙관적으로 틀리는 action 열을 골라낼 수 있다(2607.10362).

**구현**
- `src/iawm/pos_probe.py`, `scripts/fit_pos_probe.py`: z^c에서 agent와 obstacle 위치(px)를 읽는 MLP probe. base(`e32_cf`) encoder의 frame latent로 학습한다. gate 모델도 같은 encoder라 같이 쓴다.
- `cost=agent_px`(`CFJEPA.criterion`): 예측한 z^c에서 읽은 agent 위치와 goal image의 z^c에서 읽은 agent 위치의 거리(px)를 모든 step에 더한다. `sum_l2`와 같은 꼴이고 공간만 다르다. `scripts/eval.py +cost=agent_px +pos_probe=<파일>`.
- `scripts/eval.py +plan_log=<npz>`: 계획마다 고른 action 열의 예측 z^c와 goal z^c, env step마다 실제 agent 위치·target 위치·contact를 기록한다. `scripts/analyze_plan_log.py`가 episode의 첫 계획(25 step을 한 번에 실행)에서 예측 위치와 실제 위치를 비교한다.
- `scripts/merge_int_weights.py`: E39에서 받아 둔 `int_only_best.pt`와 base를 합쳐 `e39_obsbig`를 복원한다.

**설정**
- 모델: `e32_cf`(CF-JEPA), `e39_obsbig`(정답 obstacle 입력, E39). 학습은 하지 않는다.
- data: 평가 시작 상태용 `collision_tworoom`(2,000 episode, 이전과 같은 seed), probe용 `r1p`(r1x 조건, seed 555, 2,000 episode).
- 조건 4개: 모델 2 × cost 2(`sum_l2`, `agent_px`). 모두 P2, r1x, `+plan_log`, L4 한 세션에서 seed를 짝지어 잰다.
- 1단계: seed 42~49. `e39_obsbig`의 (agent_px − sum_l2) 또는 agent_px에서의 (`e39_obsbig` − `e32_cf`)가 +2%p 이상이면 2단계(seed 50~57)를 네 조건 모두 잰다.

**판정 기준 (실행 전에 정함. seed 16개 paired, +5%p 이상이고 표준오차의 3배 이상)**
- Q1 cost가 병목인가: `e39_obsbig`에서 agent_px − sum_l2.
- Q2 cost를 고치면 interaction 정보가 값을 하는가: agent_px에서 `e39_obsbig` − `e32_cf`. 통과해도 정답 obstacle 입력과 위치 label로 만든 cost라는 단서가 붙는다.
- Q3 planner가 모델이 틀리는 계획을 고르는가(sum_l2 조건의 계획 기록): 실패한 episode 중 "계획이 도착한다고 예측한"(예측 위치가 어느 block에서든 goal 16px 안) 비율이 50% 이상이면 그렇다고 본다. planner가 고른 계획에서의 5 block 예측 오차를 E39의 data 위 오차(CF-JEPA 4.8 / 15.9px, `e39_obsbig` 4.5 / 6.5px, non-contact / contact)와 비교한다.
- 한계: `agent_px` cost는 위치 label로 학습한 probe를 쓰므로 진단용이다. Q3의 k block 오차는 그 시점까지 끝나지 않은 episode에서만 잴 수 있어(성공하면 env가 reset된다) 실패 쪽으로 치우친다.
- 비용 상한: L4 세션 1개, 6 CU.

**결과**

위치 probe(`pos_probe_e32_cf.pt`, `r1p` 144,956 frame)의 held-out 오차는 agent 0.68px, obstacle 1.28px이다.

Planning 성공률(%), r1x, P2, seed 42~57(16개), L4 한 세션:

| 조건 | 평균 |
|---|---|
| `e32_cf` + `sum_l2` (CF-JEPA) | 61.8 |
| `e32_cf` + `agent_px` | 67.2 |
| `e39_obsbig` + `sum_l2` | 61.8 |
| `e39_obsbig` + `agent_px` | 70.2 |

| 비교 (paired) | 차이 ± 표준오차 | 이긴 / 진 seed | 판정 |
|---|---|---|---|
| Q1 `e39_obsbig`: agent_px − sum_l2 | +8.50 ± 2.55 | 13 / 2 | 통과 (3.3 SE) |
| Q2 agent_px: `e39_obsbig` − `e32_cf` | +3.00 ± 2.15 | 9 / 7 | 미달 |
| (참고) `e32_cf`: agent_px − sum_l2 | +5.50 ± 1.42 | 15 / 1 | 기준 이상 (3.9 SE) |
| (참고) sum_l2: `e39_obsbig` − `e32_cf` | +0.00 ± 2.24 | 9 / 7 | 차이 없음 (E39는 +3.4) |

1단계(seed 8개)에서 Q1 +4.50, Q2 +1.50이라 2단계로 넘어갔다.

Q3 (episode의 첫 계획 800개씩, goal에 닿기 전까지만 비교):

| 조건 | 성공 | 실패 중 "예측으로는 도착" | 1 block 뒤 오차 (non-contact / contact) | 5 block 뒤 오차 (non-contact / contact) |
|---|---|---|---|---|
| `e32_cf` + `sum_l2` | 61.8 | 75.2% (230/306) | 19.4 / 21.9px | 30.4 / 46.7px |
| `e39_obsbig` + `sum_l2` | 61.8 | 70.6% (216/306) | 18.8 / 19.6px | 28.1 / 42.8px |
| `e32_cf` + `agent_px` | 67.2 | 100% (262/262) | 22.9 / 27.2px | 32.4 / 50.2px |
| `e39_obsbig` + `agent_px` | 70.2 | 100% (238/238) | 25.7 / 26.9px | 29.9 / 46.5px |

- 집계 수정: 처음 자동 집계(`status_live.txt`의 `RESULT_PLAN`)는 goal에 닿은 뒤의 step도 비교했다. 성공한 env는 그 자리에 멈춘 채 기록이 이어지므로 오차가 섞인다. `analyze_plan_log.py`가 goal에 처음 닿은 step까지만 비교하도록 고쳐 로컬(CPU)에서 다시 계산했고, 위 표는 고친 값이다. "예측으로는 도착" 비율은 이 수정과 무관하다.
- 계획 시점에 현재 frame에서 읽은 agent 위치 오차는 2.9~3.3px로, 예측 오차(19~27px)보다 훨씬 작다. 오차는 probe가 아니라 예측에서 나온다.
- 첫 block에서 모델이 예측한 agent 이동 거리는 평균 30.9px(`e32_cf`, sum_l2)인데 실제는 20.2px다. 예측 이동이 env에서 불가능한 거리(speed 5 × 5 step × √2 = 35.4px 초과)인 계획이 첫 block에서 30~50%다(실제로 이 거리를 넘은 block은 1% 안팎이고, 모두 obstacle에 밀린 경우다). 둘째 block부터는 그 비율이 1~6%로 줄어든다.

**해석**
- **cost는 병목이다(Q1 통과).** z^c 거리 대신 agent 위치 거리로 계획하면 두 모델 모두 오른다(+5.5, +8.5%p). z^c에 섞인 obstacle 성분이 cost를 흐린다는 E38의 해석과 맞는다. 다만 `agent_px`는 위치 label로 학습한 probe를 쓰므로 제안 방법이 아니다.
- **cost를 고쳐도 interaction 정보의 이득은 작다(Q2 미달).** 정답 obstacle 입력 모델이 CF-JEPA보다 +3.0%p로, E39(+3.4%p)와 비슷하다.
- **planner는 모델이 틀리는 계획을 고른다(Q3, 기준 50% 이상).** 실패한 episode의 70~100%에서 모델은 "도착한다"고 예측했다. planner가 고른 계획에서의 예측 오차는 1 block 뒤에 이미 19~27px로, E39에서 수집 data의 action으로 잰 5 block 뒤 오차(4.5~15.9px)보다 크다. 정답 obstacle 입력 모델도 마찬가지라, contact 예측을 고친 이득이 이 오차에 묻힌다.
- 오차는 첫 block에 몰려 있고, 그 내용은 "agent가 실제보다 멀리 간다"는 낙관적 예측이다. 후보 원인은 둘이다. (A) CEM은 정규화한 action 공간에서 범위 제한 없이 sample하고, env는 `step()`에서 action을 [-1, 1]로 자른다(`torch.clamp`). 모델은 잘리기 전 값을 받으므로 env가 실행할 수 없는 큰 action에서 큰 이동을 예측할 수 있다. (B) 첫 계획의 과거 frame은 "시작 위치에 멈춰서 action 0"으로 만든 것이라(E26, `render_past`) 학습 data와 분포가 다르다. 이번 기록에는 고른 action 값이 없어 둘을 가르지 못했다. r1x의 budget이 30 step이라 첫 계획(25 step)이 episode를 거의 정한다.

**다음:** 가장 큰 병목은 첫 계획에서의 낙관적 예측이다. 다음 실험은 (A)를 직접 확인한다. CEM 후보를 env의 action 범위로 자른 조건과 자르지 않은 조건을 두 모델에 대해 짝지어 재고, 고른 action 값을 기록해 범위 밖 비율을 잰다. 팀과 정한 뒤 번호를 붙인다.

**비용과 운영:** 세션 `e40` 약 3시간 10분(14:09~17:20), 6.20 CU(276.82 → 270.62). 상한 6 CU를 0.2 넘었다(2단계까지 간 경우의 추정이 빠듯했다). 시작 직후 세션 안 자동 반납 재설정이 응답하지 않아 손으로 다시 걸었다. 감시 서비스가 결과 저장, 세션 종료, 종료 확인까지 마쳤다.

**결과물:** `outputs/e40/res/`(git 제외)에 log 묶음 `e40_logs.tgz`, `status_live.txt`, 조건·seed별 `*_planlog.npz`, `planlog_*.txt`(자동 집계, 수정 전), probe `stable-wm/checkpoints/pos_probe_e32_cf.pt`.

## 2026-11-17 · E41. CEM action 범위 제한 (학습 없음)

**목적:** E40에서 planner가 고른 첫 계획은 첫 block에 agent가 실제보다 멀리 간다고 예측했다(예측 31px, 실제 20px, env에서 불가능한 거리 30~50%). 후보 원인 (A)를 확인한다. CEM은 정규화한 action 공간에서 범위 없이 sample하고, env는 `step()`에서 raw action을 [-1, 1]로 자른다. 모델은 잘리기 전 action을 받으므로 env가 실행할 수 없는 action에서 큰 이동을 예측할 수 있다. 모델 입력에서 같은 자르기를 하면 이 불일치가 없어진다. 자르기는 모든 모델에 똑같이 적용하는 planning 쪽 수정이고, 제안 방법은 아니다. 이 불일치가 interaction 정보의 이득을 가리고 있었는지(Q2)가 제안 방법과 관련된 질문이다.

**구현**
- `scripts/eval.py +action_clip=true`: `model.get_cost`가 받는 후보 action을 env의 action 범위를 정규화 공간으로 옮긴 [lo, hi]로 자른다. 정규화가 affine이라 env의 clamp와 정확히 같다(로컬에서 확인, 차이 2e-8). 실행하는 action(CEM mean)은 그대로 두고 env가 자른다. planning에 넣는 과거 action도 잘린 값으로 넣는다.
- plan log에 고른 action을 함께 저장하고, 조건과 무관하게 "고른 action 중 env 범위 밖 비율"(첫 block, 전체)을 `[actions]` 줄로 찍는다.

**설정**
- 모델: `e32_cf`(CF-JEPA), `e39_obsbig`(정답 obstacle 입력). 학습하지 않는다. cost는 기존 `sum_l2`(label 없음).
- 조건 4개: 모델 2 × 자르기 2. 모두 P2, r1x, `+plan_log`, L4 한 세션에서 seed를 짝지어 잰다.
- 1단계: seed 42~49. Q1 또는 Q2의 평균이 +2%p 이상이면 2단계(seed 50~57)를 네 조건 모두 잰다.
- 계획 기록 분석에는 E40의 위치 probe(`pos_probe_e32_cf.pt`)를 쓴다.

**판정 기준 (실행 전에 정함. seed 16개 paired, +5%p 이상이고 표준오차의 3배 이상)**
- Q1 자르기가 CF-JEPA를 돕는가: `e32_cf`에서 clip − 자르지 않음.
- Q2 자르면 interaction 정보가 값을 하는가: clip에서 `e39_obsbig` − `e32_cf`. 통과해도 정답 obstacle 입력이라는 단서가 붙는다.
- Q3 원인 확인: 자르지 않은 조건에서 고른 action의 첫 block 범위 밖 비율이 10% 미만이면 (A)는 주 원인이 아니고, 첫 계획의 인공적인 과거 frame(B) 쪽을 다음에 본다. clip 조건에서 첫 block 예측 이동이 실제 이동에 가까워지는지(E40: 31 대 20px)도 함께 본다.
- 비용 상한: L4 세션 1개, 6 CU.

**결과**

Planning 성공률(%), r1x, P2, `sum_l2`, seed 42~57(16개), L4 한 세션:

| 조건 | 평균 | 고른 action 중 env 범위 밖 (첫 block / 전체) |
|---|---|---|
| `e32_cf` (CF-JEPA) | 60.8 | 49.2% / 24.2% |
| `e32_cf` + clip | 73.4 | 63.4% / 43.5% |
| `e39_obsbig` | 63.8 | 43.4% / 23.3% |
| `e39_obsbig` + clip | 71.4 | 56.6% / 41.7% |

| 비교 (paired) | 차이 ± 표준오차 | 이긴 / 진 seed | 판정 |
|---|---|---|---|
| Q1 `e32_cf`: clip − 자르지 않음 | +12.62 ± 1.82 | 15 / 0 | 통과 (6.9 SE) |
| Q2 clip: `e39_obsbig` − `e32_cf` | −2.00 ± 1.64 | 4 / 10 | 미달 (방향도 반대) |
| (참고) `e39_obsbig`: clip − 자르지 않음 | +7.62 ± 1.50 | 14 / 2 | |
| (참고) 자르지 않음: `e39_obsbig` − `e32_cf` | +3.00 ± 1.67 | 9 / 6 | |

1단계(seed 8개)에서 Q1 +11.25, Q2 −3.50이라 2단계로 넘어갔다. 자르지 않은 CF-JEPA의 seed 42 값(62)은 E40과 같다.

Q3 (episode의 첫 계획 800개씩, goal에 닿기 전까지만 비교, 위치 probe로 읽음):

| 조건 | 첫 block 예측 이동이 불가능한 거리(>35.4px) | 1 block 뒤 오차 (nc / c) | 3 block 뒤 오차 (nc / c) | 실패 중 "예측으로는 도착" |
|---|---|---|---|---|
| `e32_cf` | 31.5% | 17.6 / 22.6px | 28.8 / 38.8px | 76.4% (240/314) |
| `e32_cf` + clip | 2.8% | 4.7 / 10.0px | 19.6 / 24.4px | 37.6% (80/213) |
| `e39_obsbig` | 28.6% | 19.3 / 19.5px | 28.6 / 32.9px | 69.7% (202/290) |
| `e39_obsbig` + clip | 3.4% | 3.7 / 7.3px | 9.6 / 14.9px | 25.8% (59/229) |

- 자르지 않은 조건에서 고른 첫 block action의 43~49%가 env 범위 밖이다(기준 10%). (A)가 주 원인이다. clip 조건에서는 범위 밖 비율이 더 높지만 모델이 잘린 값으로 예측하므로 불일치가 없다.
- clip을 하면 첫 block의 불가능한 예측 이동이 31%에서 3%로, 1 block 뒤 오차가 4~5배 줄어든다.
- clip 조건에서 정답 obstacle 모델은 planner가 고른 계획에서도 contact 예측이 CF-JEPA보다 정확하다(3 block 뒤 contact 14.9 대 24.4px). 실패한 episode 대부분(170/229)은 "도착하는 계획을 찾지 못했다고 스스로 예측한" 경우이고, 그 마지막 block의 예측 goal 거리는 평균 60px다.

**해석**
- **CEM과 env의 action 범위 불일치가 E40에서 본 낙관적 예측의 주 원인이었다.** 모델 입력에서 env와 같은 자르기를 하면 CF-JEPA가 60.8에서 73.4로 오른다. 학습 없는 planning 쪽 수정이고 모든 모델에 적용되므로, 이후 Collision env 비교는 clip을 켠 상태를 기준으로 한다.
- **clip으로 예측이 정직해지자 interaction 정보는 예측에서는 이득이 보이지만 성공률로는 이어지지 않는다.** 정답 obstacle 모델은 고른 계획에서 더 정확하고, 실패할 때 "도착한다"고 착각하는 비율도 낮다(26% 대 38%). 그런데 성공률은 CF-JEPA보다 2.0%p 낮다. 정확한 모델이 도착하는 계획을 찾지 못한다고 판단한 episode가 많다. 원인 후보는 셋이다. (1) z^c 거리 cost가 obstacle 성분에 흐려진다(E40 Q1에서 정답 obstacle 모델이 위치 cost로 +8.5%p, CF-JEPA는 +5.5%p). (2) CEM(300 × 30, 5 block)이 push를 피해 도는 계획을 찾지 못한다. (3) 30 step budget 안에 실제로 도착할 수 없는 episode가 있다. CF-JEPA는 낙관적으로 틀린 계획으로 가끔 성공한다(예측 미도착인데 성공 33 대 25).

**다음:** clip을 켠 상태를 새 기준(P2 + `+action_clip=true`)으로 둔다. 후보는 (1) clip + 위치 cost로 E40 Q2를 다시 재기, (2) clip 효과를 기존 benchmark(Distractor env, 원래 TwoRoom/PushT)에서도 재기다. 팀과 정한 뒤 번호를 붙인다.

**비용과 운영:** 세션 `e41` 약 2시간 33분(18:28~21:01), 4.92 CU(270.62 → 265.70). 문제 없이 끝났다.

**결과물:** `outputs/e41/res/`(git 제외)에 `e41_logs.tgz`, `status_live.txt`, 조건·seed별 `*_planlog.npz`(고른 action 포함), `planlog_*.txt`.

## 2026-11-18 · E42. 범위 제한 + 위치 cost (학습 없음)

**목적:** E41에서 action 범위를 제한하자 정답 obstacle 모델은 planner가 고른 계획에서 CF-JEPA보다 정확하게 예측했지만, 성공률은 −2.0 ± 1.6%p였다. E37의 정답 simulator 상한에서 interaction 지식의 이득은 25 step을 한 번에 실행하는 지금의 protocol에서 크고(blind 70.0 대 aware 96.7), 자주 다시 계획하면 거의 사라진다(blind 95.3). 그래서 이 protocol에서 정확한 예측이 성공률로 이어지지 않는 이유를 가른다. 후보는 cost(z^c 거리에 obstacle 성분이 섞인다, E40)와 긴 구간 예측 오차(5 block 뒤 13~25px, E41)다. 위치 cost는 정답 simulator가 쓰는 "실제 위치 거리"에 가까우므로, 이것으로 바꿨을 때 차이가 생기면 cost가 병목이다.

**설정**
- 모델: `e32_cf`(CF-JEPA), `e39_obsbig`(정답 obstacle 입력). 학습하지 않는다.
- 조건 4개: 모델 2 × cost 2(`sum_l2`, `agent_px`), 모두 `+action_clip=true`, P2, r1x, `+plan_log`. L4 한 세션에서 seed를 짝지어 잰다. 위치 probe는 E40의 `pos_probe_e32_cf.pt`.
- 1단계: seed 42~49. Q1의 평균이 +2%p 이상이거나 Q2 중 하나가 +2%p 이상이면 2단계(seed 50~57)를 네 조건 모두 잰다.

**판정 기준 (실행 전에 정함. seed 16개 paired, +5%p 이상이고 표준오차의 3배 이상)**
- Q1 (주 질문): `agent_px`에서 `e39_obsbig` − `e32_cf`. 통과하면 cost가 병목이고, label 없는 위치 cost(E43)로 넘어간다. 미달이면 남은 한계는 긴 구간 예측 오차이고, E43은 하지 않는다.
- Q2 (참고): 모델별 `agent_px` − `sum_l2`. E40(clip 없음)의 +5.5(CF-JEPA), +8.5(정답 obstacle)와 비교한다.
- Q3 (참고): 실패한 episode 중 "예측으로는 도착"과 "예측 미도착"의 비율. 위치 cost에서 정답 obstacle 모델의 "예측 미도착" 실패가 줄어드는지 본다.
- 한계: `agent_px`는 위치 label로 학습한 probe를 쓰므로 진단용이다.
- 비용 상한: L4 세션 1개, 6 CU.

**결과:** (실행 후 적는다)

## 2026-11-18 · E44. 기존 benchmark(TwoRoom)에서 CEM action 범위 제한 (학습 없음)

**목적:** E41의 action 범위 불일치(CEM은 범위 없이 sample, env는 [-1, 1]로 자른다)는 Collision env만의 문제가 아니라 stable-worldmodel의 TwoRoom 계열 전체에 있다. 기존 benchmark에서도 범위 제한이 성공률을 바꾸는지 잰다. 수업 피드백("기존 benchmark task의 success rate를 함께 제시")에 해당한다. PushT env는 action을 자르지 않고 그대로 실행하므로(`stable_worldmodel/envs/pusht/env.py`의 `step`) 같은 불일치가 없어 제외한다.

**설정**
- CF-JEPA: `cfjepa_tworoom_n2`(E3, Distractor TwoRoom N=2로 학습), Distractor TwoRoom N=2에서 평가(논문 Table III 조건). cost는 논문대로 `sum_l2`.
- LeWM: 공개 checkpoint `lewm-tworooms`(E0), 원래 TwoRoom(N=0)에서 평가(논문 Table II 조건).
- 조건 4개: 모델 2 × 범위 제한 없음/있음. 평가 설정은 E0, E3과 같다(CEM 300 × 30, 50 episode, budget 50). L4 한 세션에서 seed를 짝지어 잰다.
- 1단계: seed 42~49. Q1 또는 Q2의 평균이 +2%p 이상이면 2단계(seed 50~57)를 잰다. 둘 다 미달이면 seed 8개로 끝낸다.

**판정 기준 (실행 전에 정함. paired, +5%p 이상이고 표준오차의 3배 이상)**
- Q1: CF-JEPA(N=2)에서 범위 제한 − 없음.
- Q2: LeWM(N=0)에서 범위 제한 − 없음.
- 참고: 고른 action 중 env 범위 밖 비율(첫 block, 전체). 기존 성공률이 높아(0.86~0.94) 오를 여지가 6~14%p뿐이므로, 미달이어도 범위 밖 비율과 함께 "이 불일치가 충돌이 있는 환경에서 커진다"로 해석할 수 있다.
- 비용 상한: L4 세션 1개, 6 CU.

**결과:** (실행 후 적는다)

## 2026-11-18 · E43. label 없는 위치 cost: action만으로 학습한 위치 probe

**목적:** E42에서 위치 cost(정답 위치 label로 학습한 probe)를 쓰면 정답 obstacle 모델이 CF-JEPA를 앞서는지 잰다. 이 실험은 그 위치 cost를 label 없이 만든다. TwoRoom에서 agent는 한 step에 `clip(a, -1, 1) × speed`만큼 움직인다(벽과 obstacle에 막히거나 밀리지 않으면). 그래서 같은 episode의 두 frame 사이 이동은 data에 이미 있는 action의 누적합으로 정해진다. z^c → 2차원 위치 MLP f를 `f(z_{t+k}) − f(z_t) ≈ speed · Σ clip(a_i)`로 학습한다. 벽과 push 구간은 Huber loss로 영향을 줄이고, contact label도 쓰지 않는다. 학습에 쓰는 것은 frame과 action뿐이므로 CF-JEPA의 학습 가정(위치 label 없음)을 지킨다. E42 Q1이 통과할 때만 실행한다.

**구현:** `scripts/fit_odom_probe.py`. 저장 형식은 `iawm.pos_probe`와 같아 `+cost=agent_px +pos_probe=odom_probe_e32_cf.pt`로 쓴다. 정답 위치는 held-out 보고(disp: 같은 episode 두 frame의 이동 오차, abs: 평행 이동 하나만 맞춘 위치 오차, affine: 참고)에만 쓴다. 벽이 있는 합성 random walk에서 위치를 0.2px로 복원하는 것을 로컬에서 확인했다.

**설정**
- probe 학습: `e32_cf` encoder의 frame latent, `r1p`(r1x 조건 2,000 episode, E40과 같은 data). 학습 전에 기준을 정한다: held-out abs 오차가 10px을 넘으면 planning 평가를 하지 않고 멈춘다.
- 조건 4개, 모두 P2 + `+action_clip=true`, r1x, `+plan_log`, L4 한 세션 paired: `e32_cf` + `sum_l2`(CF-JEPA 기준), `e32_cf` + odom cost, `e39_obsbig` + odom cost, `e39_obsbig` + 정답 위치 cost(참고, label 상한).
- 1단계 seed 42~49, Q1 또는 Q2가 +2%p 이상이면 2단계 seed 50~57.

**판정 기준 (실행 전에 정함. seed 16개 paired, +5%p 이상이고 표준오차의 3배 이상)**
- Q1 (주 질문, 공정 비교): odom cost에서 `e39_obsbig` − `e32_cf`. 같은 label 없는 cost를 쓴 두 모델의 차이다.
- Q2 (최종 비교): `e39_obsbig` + odom cost − `e32_cf` + `sum_l2`. 제안 구성 전체와 CF-JEPA(범위 제한 포함) 그대로의 차이다.
- 참고: `e32_cf`에서 odom − `sum_l2`, `e39_obsbig`에서 정답 위치 cost − odom cost(label 없는 cost가 label cost를 얼마나 따라가는가).
- 단서: 통과해도 obstacle 입력은 정답(E39의 `obs_oracle`)이다. obstacle 상태를 화면에서 읽는 부분은 남은 과제다.
- 비용 상한: L4 세션 1개, 6 CU.

**결과:** (실행 후 적는다)
