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
- 코드는 `yeonwoo/midterm-figs` branch의 `scripts/colab/eval_obstacle_sweep.sh`, `scripts/plot_obstacle_sweep.py`다(main에는 아직 합치지 않았다).

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
- 코드는 `yeonwoo/midterm-figs` branch의 `scripts/record_gate_rollout.py`다. 화면 아래에 α_t 선과 실제 contact 구간(띠)을 함께 그린다.

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
