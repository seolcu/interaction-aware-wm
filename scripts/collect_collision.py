"""Collision TwoRoom 학습 data 수집.

원래 TwoRoom data(`quentinll/lewm-tworooms`)와 같은 방식으로 모은다. 환경만 바꾼다.
- 정책: stable-worldmodel `scripts/data/collect_tworooms.py`와 같은 ExpertPolicy
  (action_noise=2.0, action_repeat_prob=0.05), episode 최대 100 step, 224×224.
- 형식: HDF5 (`$STABLEWM_HOME/datasets/<name>.h5`). 원래 tworoom.h5와 column이 같고,
  env info에서 `obstacle_pos`, `obstacle_vel`, `contact`, `pinned`가 더 들어간다.
  이 네 column은 학습에 쓰지 않고 분석(충돌/비충돌 구간 오차, α AUROC)의 정답으로 쓴다.
- 저장 방식: World.collect는 압축 없이 쓴다(frame당 150KB). 원래 tworoom.h5처럼 pixels를
  blosc(lz4, level 5, shuffle, chunk 100 frame)로 다시 써서 크기를 1/15 정도로 줄인다.
  원래 data에 있는 `ep_idx` column도 이때 채운다(평가 script가 쓴다).
- seed를 고정하면 같은 data가 다시 만들어진다. 그래서 data 파일 대신 이 script와 seed를 공유한다.

사용:
    python scripts/collect_collision.py                      # 1만 episode, obstacle 1개
    python scripts/collect_collision.py --episodes 200 --name collision_debug
    python scripts/collect_collision.py --num-obstacles 3 --name collision_tworoom_k3
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

import numpy as np  # noqa: E402
import stable_worldmodel as swm  # noqa: E402
from stable_worldmodel.envs.two_room import ExpertPolicy  # noqa: E402

import envs.collision_tworoom  # noqa: E402,F401  (gym 등록)


def finalize(raw: Path, out: Path, block: int = 10000):
    """raw h5를 원래 tworoom.h5와 같은 압축 형식으로 다시 쓰고 ep_idx를 더한다."""
    import h5py
    import hdf5plugin

    blosc = hdf5plugin.Blosc(cname='lz4', clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)
    with h5py.File(raw, 'r') as src, h5py.File(out, 'w') as dst:
        ep_len = src['ep_len'][:]
        n = int(ep_len.sum())
        for k in src.keys():
            d = src[k]
            if d.shape[0] != n:  # ep_len, ep_offset
                dst.create_dataset(k, data=d[:])
                continue
            chunk_rows = 100 if k.startswith('pixels') else 1000
            o = dst.create_dataset(
                k, shape=d.shape, dtype=d.dtype,
                chunks=(min(chunk_rows, n), *d.shape[1:]), **blosc,
            )
            for i in range(0, n, block):
                o[i:i + block] = d[i:i + block]
        if 'ep_idx' not in src:
            dst.create_dataset('ep_idx', data=np.repeat(np.arange(len(ep_len), dtype=np.int32), ep_len))
    raw.unlink()


def summarize(path: Path):
    import h5py

    with h5py.File(path, 'r') as f:
        ep_len = f['ep_len'][:]
        contact = f['contact'][:].astype(bool)
        pinned = f['pinned'][:].astype(bool)
        ep_idx = f['ep_idx'][:]
        term = f['terminated'][:].astype(bool)
        print(f'columns: {sorted(f.keys())}')
    ep_contact = np.bincount(ep_idx, weights=contact, minlength=len(ep_len)) > 0
    ep_success = np.bincount(ep_idx, weights=term, minlength=len(ep_len)) > 0
    stats = {
        'episodes': int(len(ep_len)),
        'frames': int(ep_len.sum()),
        'mean_ep_len': float(ep_len.mean()),
        'contact_step_ratio': float(contact.mean()),
        'pinned_step_ratio': float(pinned.mean()),
        'episodes_with_contact': float(ep_contact.mean()),
        'episodes_reaching_goal': float(ep_success.mean()),
    }
    for k, v in stats.items():
        print(f'{k}: {v:.4f}' if isinstance(v, float) else f'{k}: {v}')
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=10000)
    ap.add_argument('--num-envs', type=int, default=10)
    ap.add_argument('--seed', type=int, default=3072)  # 원래 TwoRoom 수집 config와 같은 값
    ap.add_argument('--num-obstacles', type=int, default=1)
    ap.add_argument('--spawn', default='crossing', choices=['crossing', 'random'])
    ap.add_argument('--obstacle-radius', type=float, default=14.0)  # E27: 평가 조건 r3b(22)와 맞춘 data
    ap.add_argument('--name', default='collision_tworoom')
    args = ap.parse_args()

    world = swm.World(
        'iawm/CollisionTwoRoom-v1',
        num_envs=args.num_envs,
        image_shape=(224, 224),
        max_episode_steps=100,
        num_obstacles=args.num_obstacles,
        spawn_mode=args.spawn,
        obstacle_radius=args.obstacle_radius,
    )
    world.set_policy(ExpertPolicy(action_noise=2.0, action_repeat_prob=0.05, seed=args.seed))

    out = Path(swm.data.utils.get_cache_dir()) / 'datasets' / f'{args.name}.h5'
    raw = out.with_name(f'{args.name}.raw.h5')
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    world.collect(raw, episodes=args.episodes, seed=int(rng.integers(0, 1_000_000)), format='hdf5')
    print(f'collected {raw} ({raw.stat().st_size / 1e9:.2f} GB, {time.time() - t0:.0f}s)')
    finalize(raw, out)
    print(f'saved {out} ({out.stat().st_size / 1e9:.2f} GB, {time.time() - t0:.0f}s)')
    summarize(out)


if __name__ == '__main__':
    main()
