# Experiment Log

실험마다 날짜, 목적, 실행 방법, 결과, 남은 문제를 적는다. dataset, checkpoint, 영상은 git에 넣지 않는다.

## 2026-10-11 · E0. LeWM TwoRoom baseline 재현

**목적:** 공개 LeWM TwoRoom checkpoint가 우리 Colab 환경에서 논문 수치를 재현하는지 확인한다. 이후 CF-JEPA 재구현과 Env. ②의 출발점이다.

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

**다음:** 같은 pipeline에 Env. ② obstacle을 넣는다(B). CF-JEPA 재구현의 기준선으로 이 0.86을 쓴다.

## 2026-10-11 · E1. Env. ② collision TwoRoom prototype

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
