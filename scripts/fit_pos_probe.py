"""z^c에서 agent와 obstacle 위치(px)를 읽는 probe를 학습해 저장한다 (E40).

base 모델의 encoder로 data의 모든 frame을 encoding하고(latent cache를 쓴다), z^c -> [agent xy, obstacle xy] MLP를
학습한다. 저장한 probe는 `scripts/eval.py +cost=agent_px +pos_probe=<파일>`과 `analyze_plan_log.py`가 쓴다.
gate 모델도 같은 base encoder를 쓰므로 probe 하나를 같이 쓴다.

사용:
    python scripts/fit_pos_probe.py --base e32_cf/weights_epoch_10.pt --data r1p.h5 --out pos_probe_e32_cf.pt
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT), str(ROOT / 'scripts')]

import numpy as np

from multistep_px import positions
from train_int_frozen import frame_latents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True, help='$STABLEWM_HOME/checkpoints 아래 파일 이름')
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    import h5py
    import hdf5plugin  # noqa: F401
    import stable_worldmodel as swm

    from iawm import pos_probe

    device = 'cuda'
    home = Path(swm.data.utils.get_cache_dir())
    model = swm.wm.utils.load_pretrained(args.base).to(device).eval()
    tag = args.base.replace('/', '_').replace('.pt', '')
    with h5py.File(home / 'datasets' / args.data, 'r') as f:
        z = frame_latents(model, f, device, home / 'latent_cache' / f'{tag}__{Path(args.data).stem}.npy')
        pos = positions(f)
    probe, (pa, po) = pos_probe.fit(z[:, :model.dim_c].astype(np.float32), pos, device, steps=args.steps, seed=args.seed)
    pos_probe.save(probe, home / 'checkpoints' / args.out)
    print(f'RESULT_PROBE {args.out} frames {len(z)} heldout_agent_px {pa:.3f} heldout_obstacle_px {po:.3f}')


if __name__ == '__main__':
    main()
