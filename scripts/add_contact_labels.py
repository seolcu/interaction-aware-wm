"""Collision TwoRoom h5에 transition 단위 contact 라벨 `contact_next`를 더한다 (E9, supervised gate용).

학습 data loader는 action 외의 column을 frameskip 간격으로 건너뛰어 읽는다(row s, s+5, s+10, ...).
그래서 `contact` column을 그대로 읽으면 frame 사이 5 step 중 1 step의 contact만 보인다.
이 script는 row j에 "frame j -> j+frameskip transition 동안 contact가 있었는가"를 미리 계산해 둔다.

    contact_next[j] = any(contact[j+1 .. j+frameskip])   (같은 episode 안에서만)

World.collect는 row j에 step j-1 이후의 frame과 그 step의 info(contact)를 저장하므로,
frame j -> j+frameskip 동안의 contact는 row j+1 .. j+frameskip이다(scripts/analyze_gate.py와 같은 정의).

기존 column은 건드리지 않고 `contact_next` (float32, shape (N, 1))만 추가한다.

원본을 학습 중인 다른 process가 읽고 있을 수 있으므로, 사본을 만들어 거기에 쓴다
(configs/train/data/collision_tworoom_contact.yaml이 이 사본을 읽는다).

`--label-frac p`를 주면 episode의 p 비율에만 라벨을 남긴 column `contact_next_p<백분율>`
(예: 0.02 -> `contact_next_p2`)도 더한다. 나머지 episode는 NaN이고, 학습의 contact 보조 loss가
NaN을 건너뛴다. "적은 라벨" 실험(E10)용이다. 어떤 episode에 라벨을 남길지는 --label-seed로 고정한다.

사용:
    cp $STABLEWM_HOME/datasets/collision_tworoom.h5 $STABLEWM_HOME/datasets/collision_tworoom_contact.h5
    python scripts/add_contact_labels.py --data collision_tworoom_contact.h5
    python scripts/add_contact_labels.py --data collision_tworoom_contact.h5 --label-frac 0.02
"""

import argparse
from pathlib import Path

import numpy as np


def contact_next_labels(contact: np.ndarray, ep_off: np.ndarray, ep_len: np.ndarray, frameskip: int) -> np.ndarray:
    out = np.zeros(len(contact), dtype=np.float32)
    for o, L in zip(ep_off, ep_len):
        c = contact[o:o + L].astype(bool)
        # cs[k] = c[0..k-1]의 합. row j의 창 c[j+1 .. j+fs]는 cs[min(j+fs+1, L)] - cs[j+1]이다.
        cs = np.concatenate([[0], np.cumsum(c)])
        j = np.arange(L)
        hi = np.minimum(j + frameskip + 1, L)
        lo = np.minimum(j + 1, L)
        out[o:o + L] = (cs[hi] - cs[lo] > 0).astype(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='collision_tworoom_contact.h5')
    ap.add_argument('--frameskip', type=int, default=5)
    ap.add_argument('--force', action='store_true', help='contact_next가 이미 있으면 다시 쓴다')
    ap.add_argument('--label-frac', type=float, default=None, help='이 비율의 episode에만 라벨을 남긴 column도 만든다')
    ap.add_argument('--label-seed', type=int, default=0)
    args = ap.parse_args()

    import h5py
    import hdf5plugin
    import stable_worldmodel as swm

    path = Path(swm.data.utils.get_cache_dir()) / 'datasets' / args.data
    blosc = hdf5plugin.Blosc(cname='lz4', clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)
    with h5py.File(path, 'a') as f:
        ep_len = f['ep_len'][:].astype(np.int64)
        ep_off = f['ep_offset'][:].astype(np.int64) if 'ep_offset' in f else np.concatenate([[0], np.cumsum(ep_len)[:-1]])
        contact = f['contact'][:].reshape(len(f['contact']), -1)[:, 0]
        lab = contact_next_labels(contact, ep_off, ep_len, args.frameskip)
        if 'contact_next' in f and not args.force:
            print(f'{path}: contact_next가 이미 있다. 다시 쓰려면 --force')
        else:
            if 'contact_next' in f:
                del f['contact_next']
            f.create_dataset('contact_next', data=lab[:, None], chunks=(min(1000, len(lab)), 1), **blosc)
            print(f'{path}: contact_next 추가 (rows {len(lab)}, frameskip {args.frameskip})')
            print(f'contact step ratio {contact.astype(bool).mean():.4f}  contact_next ratio {lab.mean():.4f}')
        if args.label_frac is not None:
            name = f'contact_next_p{round(args.label_frac * 100):d}'
            rng = np.random.default_rng(args.label_seed)
            keep = rng.random(len(ep_len)) < args.label_frac
            part = np.full(len(lab), np.nan, dtype=np.float32)
            for o, L, k in zip(ep_off, ep_len, keep):
                if k:
                    part[o:o + L] = lab[o:o + L]
            if name in f:
                del f[name]
            f.create_dataset(name, data=part[:, None], chunks=(min(1000, len(part)), 1), **blosc)
            n_lab = int((~np.isnan(part)).sum())
            print(f'{name} 추가: 라벨 episode {keep.sum()}/{len(ep_len)}, 라벨 row {n_lab}/{len(part)} ({n_lab / len(part):.4f})')

if __name__ == '__main__':
    main()
