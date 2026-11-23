"""E39 smoke test: iawm.obs_oracle이 계산한 obstacle 상태가 실제 env 궤적과 맞는지 확인한다.

planning에 넣는 정답 obstacle 상태는 세 군데에서 나온다. 셋이 어긋나면 h_int가 학습 때와 다른 값을 받는다.
1. 미래: `future_states`가 env에서 obstacle만 앞으로 돌린 값 == 실제로 step을 밟은 뒤의 상태
2. 과거: `past_state`가 속도를 뒤집어 되돌린 값 == 실제로 그 step에 있던 상태 (반사 포함)
3. 학습 data: h5의 row j에 저장된 위치와 속도에서 env의 이동 규칙을 한 번 적용하면 row j+1이 나오는지 (--data를 주면 확인)

사용: python scripts/colab/check_obs_oracle.py [--data r1dbg.h5]
성공하면 마지막 줄에 "OBS_ORACLE CHECK OK"를 찍는다.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

import numpy as np

from envs.collision_tworoom import CollisionTwoRoomEnv
from iawm import obs_oracle as oo

FS, TOL = 5, 1e-2


def env_check(seeds=20, steps=60, radius=30.0):
    err_f, err_p = [], []
    for seed in range(seeds):
        e = CollisionTwoRoomEnv(num_obstacles=1, obstacle_radius=radius)
        e.reset(seed=seed)
        rng = np.random.default_rng(seed)
        states, fut = [oo.current_state(e)], {}
        for t in range(steps):
            if t % FS == 0:
                fut[t] = oo.future_states(e, 3, FS)
            for k in (FS, 2 * FS):
                if t >= k:
                    p, v = oo.past_state(e, k)
                    err_p.append((t - k, max(np.abs(p - states[t - k][0]).max(), np.abs(v - states[t - k][1]).max())))
            before = oo.current_state(e)
            _ = oo.future_states(e, 1, FS)  # 호출이 env 상태를 바꾸지 않는지
            after = oo.current_state(e)
            assert np.array_equal(before[0], after[0]) and np.array_equal(before[1], after[1]), 'env state changed'
            e.step(rng.uniform(-1, 1, 2).astype(np.float32))
            states.append(oo.current_state(e))
        for t, fs_ in fut.items():
            for b, (p, v) in enumerate(fs_):
                tt = t + FS * (b + 1)
                if tt < len(states):
                    err_f.append(max(np.abs(p - states[tt][0]).max(), np.abs(v - states[tt][1]).max()))
        # planning 입력 모양: history 3 + horizon 5 -> 7 frame, 각 frame의 pos_next는 다음 frame의 pos
        hist = [states[-1 - FS * j] for j in (2, 1, 0)]
        po = oo.plan_oracle(hist, e, 5, FS)
        assert po.shape == (7, 6), po.shape
        assert np.allclose(po[:-1, 4:], po[1:, :2]), 'pos_next mismatch'
    err_f = np.array(err_f)
    idx, err_p = np.array([i for i, _ in err_p]), np.array([x for _, x in err_p])
    # 되돌려서 닿는 시점이 episode의 첫 frame(index 0)이면 따로 센다. spawn은 obstacle을 방 경계에 딱 붙여 놓을 수 있고,
    # 그 위치에서는 되돌릴 때 반사 판정이 반올림에 따라 갈려 속도 부호가 뒤집힐 수 있다. planning에서는 문제가 아니다:
    # 시작 전 frame(render_past)과 그 상태(past_state)가 같은 계산에서 나오므로 서로 맞는다.
    first, rest = err_p[idx == 0], err_p[idx > 0]
    print(f'future: n={len(err_f)} max err {err_f.max():.5f} frac>{TOL} {(err_f > TOL).mean():.4f}')
    print(f'past (to frame >= 1): n={len(rest)} max err {rest.max():.5f} frac>{TOL} {(rest > TOL).mean():.4f}')
    print(f'past (to frame 0, spawn on a room boundary is possible): n={len(first)} mismatches {(first > TOL).sum()}')
    assert (err_f > TOL).mean() == 0, 'future obstacle states do not match the env'
    assert (rest > TOL).mean() == 0, 'past obstacle states do not match the env'


def data_check(name, radius=30.0):
    import h5py
    import hdf5plugin  # noqa: F401
    import stable_worldmodel as swm

    path = Path(swm.data.utils.get_cache_dir()) / 'datasets' / name
    with h5py.File(path, 'r') as f:
        n = len(f['obstacle_pos'])
        pos = f['obstacle_pos'][:].reshape(n, -1).astype(np.float32)
        vel = f['obstacle_vel'][:].reshape(n, -1).astype(np.float32)
        ep_len = f['ep_len'][:].astype(np.int64)
        col = oo.dataset_column(f, FS)
    off = np.concatenate([[0], np.cumsum(ep_len)[:-1]])
    same = np.ones(n, bool)
    same[off[1:] - 1] = False  # episode 마지막 row는 다음 row가 다른 episode다
    same[-1] = False
    j = np.nonzero(same)[0]
    # row j의 (위치, 속도)에서 env의 obstacle 이동(반사 포함)을 한 번 하면 row j+1의 (위치, 속도)가 나와야 한다.
    # 맞으면 "row j의 속도 = 다음 step에 쓰이는 속도"이고, env의 방 크기가 data를 모을 때와 같다는 뜻이다.
    import torch
    e = CollisionTwoRoomEnv(num_obstacles=pos.shape[1] // 2, obstacle_radius=radius)
    e.reset(seed=0)
    rng = np.random.default_rng(0)
    pick = rng.choice(j, size=min(20000, len(j)), replace=False)
    ok = 0
    for r in pick:
        e.obstacle_pos = torch.from_numpy(pos[r].reshape(-1, 2).copy())
        e.obstacle_vel = torch.from_numpy(vel[r].reshape(-1, 2).copy())
        e._move_obstacles()
        p1, v1 = oo.current_state(e)
        ok += (np.abs(p1 - pos[r + 1]).max() < TOL) and (np.abs(v1 - vel[r + 1]).max() < TOL)
    frac = ok / len(pick)
    refl = (np.abs(vel[j + 1] - vel[j]).max(1) > TOL).mean()
    print(f'data {name}: rows {n}, one env move from row j reproduces row j+1 in {frac:.4f} of {len(pick)} rows '
          f'(reflection rows {refl:.4f}); column {col.shape} pos range [{col[:, :2].min():.2f}, {col[:, :2].max():.2f}] '
          f'vel range [{col[:, 2:4].min():.2f}, {col[:, 2:4].max():.2f}]')
    assert frac > 0.99, 'h5의 obstacle_pos / obstacle_vel이 env의 이동 규칙과 맞지 않는다: obs_oracle의 가정을 다시 확인한다'
    assert np.isfinite(col).all() and np.abs(col).max() < 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=None)
    ap.add_argument('--radius', type=float, default=30.0)
    args = ap.parse_args()
    env_check()
    if args.data:
        data_check(args.data, args.radius)
    print('OBS_ORACLE CHECK OK')


if __name__ == '__main__':
    main()
