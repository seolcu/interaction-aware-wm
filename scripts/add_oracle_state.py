"""Collision TwoRoom h5에 frame 단위 정답 state column `oracle_state`를 더한다 (E16 oracle h_int 진단).

oracle_state[j] = [agent 위치(2), obstacle 위치(2K), obstacle 속도(2K), obstacle - agent(2K)]  (K = obstacle 수)

학습 data loader는 frameskip 간격의 frame row를 읽으므로, row j의 값이 그 frame의 state가 된다.
train.py가 다른 column처럼 z-score 정규화한다. 기존 column은 건드리지 않는다.

사용:
    python scripts/add_oracle_state.py --data collision_tworoom_contact.h5
"""

import argparse
from pathlib import Path

import numpy as np


def oracle_state(agent: np.ndarray, obs: np.ndarray, vel: np.ndarray) -> np.ndarray:
    n = len(agent)
    rel = (obs.reshape(n, -1, 2) - agent[:, None, :]).reshape(n, -1)
    return np.concatenate([agent, obs, vel, rel], 1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='collision_tworoom_contact.h5')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    import h5py
    import hdf5plugin
    import stable_worldmodel as swm

    path = Path(swm.data.utils.get_cache_dir()) / 'datasets' / args.data
    blosc = hdf5plugin.Blosc(cname='lz4', clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)
    with h5py.File(path, 'a') as f:
        if 'oracle_state' in f and not args.force:
            print(f'{path}: oracle_state가 이미 있다. 다시 쓰려면 --force')
            return
        n = len(f['obstacle_pos'])
        key = 'pos_agent' if 'pos_agent' in f else 'proprio'
        agent = f[key][:].reshape(n, -1)[:, :2].astype(np.float32)
        obs = f['obstacle_pos'][:].reshape(n, -1).astype(np.float32)
        vel = f['obstacle_vel'][:].reshape(n, -1).astype(np.float32)
        st = np.nan_to_num(oracle_state(agent, obs, vel), nan=0.0)
        if 'oracle_state' in f:
            del f['oracle_state']
        f.create_dataset('oracle_state', data=st, chunks=(min(1000, n), st.shape[1]), **blosc)
        print(f'{path}: oracle_state 추가 (rows {n}, dim {st.shape[1]}, agent from {key})')


if __name__ == '__main__':
    main()
