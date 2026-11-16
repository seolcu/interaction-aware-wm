"""정답 obstacle 상태를 h_int 입력으로 넣을 때의 형식 (E39 상한 실험).

obstacle은 action과 agent의 영향을 받지 않고 움직인다(envs/collision_tworoom.py). 그래서 planning 중에도
env 상태에서 미래 obstacle 궤적을 정확히 계산할 수 있다. "z^u가 obstacle을 완벽하게 담고 완벽하게 예측한다면
gate 구조가 planning을 얼마나 올리는가"를 재는 데 쓴다. 실제 모델이 아니라 진단용이다.

한 frame의 값 (obstacle K개면 6K 차원):
    [pos_t (2K), vel_t (2K), pos_{t+frameskip} (2K)]
pos는 (px - 112) / 112, vel은 (px/step) / 6으로 고정 scale한다. dataset 통계에 의존하지 않으므로
학습 data와 planning에서 값이 정확히 같다.

- `dataset_column`: h5의 obstacle_pos, obstacle_vel에서 row별 값을 만든다(학습, 분석).
- `plan_states`, `past_state`: planning 중 env에서 현재와 미래, 시작 전 상태를 계산한다(scripts/eval.py).
"""

from __future__ import annotations

import numpy as np

KEY = 'obs_oracle'
POS_CENTER, POS_SCALE, VEL_SCALE = 112.0, 112.0, 6.0


def encode(pos, vel, pos_next) -> np.ndarray:
    """pos, vel, pos_next: (..., 2K) px 단위. 반환: (..., 6K) float32."""
    pos, vel, pos_next = (np.asarray(x, dtype=np.float32) for x in (pos, vel, pos_next))
    return np.concatenate(
        [(pos - POS_CENTER) / POS_SCALE, vel / VEL_SCALE, (pos_next - POS_CENTER) / POS_SCALE], axis=-1
    ).astype(np.float32)


def dataset_column(f, frameskip: int = 5) -> np.ndarray:
    """h5 file 전체 row에 대한 값 (N, 6K).

    World.collect는 row j에 frame j와 그 시점의 obstacle 위치, 다음 step에 쓰일 속도를 저장한다.
    pos_next는 같은 episode의 row j+frameskip 위치다. episode 끝 frameskip row는 다음 frame이 없어
    pos + frameskip·vel로 채운다(이 row들은 clip의 마지막 frame으로만 쓰여 입력으로 들어가지 않는다).
    """
    n = len(f['obstacle_pos'])
    pos = np.nan_to_num(f['obstacle_pos'][:].reshape(n, -1).astype(np.float32))
    vel = np.nan_to_num(f['obstacle_vel'][:].reshape(n, -1).astype(np.float32))
    ep_len = f['ep_len'][:].astype(np.int64)
    off = f['ep_offset'][:].astype(np.int64) if 'ep_offset' in f else np.concatenate([[0], np.cumsum(ep_len)[:-1]])
    nxt = pos + frameskip * vel
    for o, L in zip(off, ep_len):
        if L > frameskip:
            nxt[o:o + L - frameskip] = pos[o + frameskip:o + L]
    return encode(pos, vel, nxt)


def _flat(t) -> np.ndarray:
    return t.detach().cpu().numpy().astype(np.float32).reshape(-1).copy()


def current_state(e) -> tuple[np.ndarray, np.ndarray]:
    """지금 env의 obstacle (pos, vel), 각각 (2K,)."""
    return _flat(e.obstacle_pos), _flat(e.obstacle_vel)


def past_state(e, steps_back: int) -> tuple[np.ndarray, np.ndarray]:
    """episode 시작 `steps_back` step 전의 obstacle (pos, vel). CollisionTwoRoomEnv.render_past와 같은 계산이다.

    속도를 뒤집어 거꾸로 움직이면 반사까지 포함해 원래 궤적을 되짚는다. 그 시점의 앞으로 가는 속도는
    거꾸로 가던 속도의 부호를 뒤집은 값이다. env 상태는 그대로 둔다.
    """
    pos0, vel0 = e.obstacle_pos.clone(), e.obstacle_vel.clone()
    try:
        e.obstacle_vel = -e.obstacle_vel
        for _ in range(steps_back):
            e._move_obstacles()
        return _flat(e.obstacle_pos), -_flat(e.obstacle_vel)
    finally:
        e.obstacle_pos, e.obstacle_vel = pos0, vel0


def future_states(e, blocks: int, frameskip: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """지금부터 frameskip, 2·frameskip, ..., blocks·frameskip step 뒤의 obstacle (pos, vel). env 상태는 그대로 둔다."""
    pos0, vel0 = e.obstacle_pos.clone(), e.obstacle_vel.clone()
    out = []
    try:
        for _ in range(blocks):
            for _ in range(frameskip):
                e._move_obstacles()
            out.append((_flat(e.obstacle_pos), _flat(e.obstacle_vel)))
    finally:
        e.obstacle_pos, e.obstacle_vel = pos0, vel0
    return out


def plan_oracle(history_states: list[tuple[np.ndarray, np.ndarray]], e, horizon: int, frameskip: int) -> np.ndarray:
    """planning rollout에 넣을 값 (H + horizon - 1, 6K).

    history_states: planning에 넣는 history frame H장(마지막이 지금)의 (pos, vel).
    rollout은 frame 0..H-1(history)에서 시작해 horizon번 예측한다. k번째 예측의 입력 window는 frame
    k..k+H-1이므로, frame 0..H+horizon-2의 값이 필요하다. 각 frame의 pos_next는 다음 frame의 pos다.
    """
    states = list(history_states) + future_states(e, horizon, frameskip)
    pos = np.stack([s[0] for s in states])
    vel = np.stack([s[1] for s in states])
    return encode(pos[:-1], vel[:-1], pos[1:])
