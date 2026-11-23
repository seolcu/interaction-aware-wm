"""base(CF-JEPA) weights와 학습한 interaction module만의 weights를 합쳐 gate 모델 checkpoint를 만든다 (E40).

`train_int_frozen.py`는 전체 weights(약 80MB)와 함께 학습한 부분만 담은 `int_only_best.pt`(수 MB)를 남긴다.
세션에서는 작은 파일만 받아 두고, 다음 세션에서 base와 합쳐 복원한다.

사용:
    python scripts/merge_int_weights.py --base e32_cf/weights_epoch_10.pt --dir e39_obsbig
    ($STABLEWM_HOME/checkpoints/e39_obsbig/ 에 config.json과 int_only_best.pt가 있어야 한다. weights_best.pt를 쓴다.)
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--dir', required=True)
    args = ap.parse_args()

    import hydra
    import stable_worldmodel as swm

    ck = Path(swm.data.utils.get_cache_dir()) / 'checkpoints'
    d = ck / args.dir
    model = hydra.utils.instantiate(json.load(open(d / 'config.json')))
    sd = torch.load(ck / args.base, map_location='cpu', weights_only=True)
    part = torch.load(d / 'int_only_best.pt', map_location='cpu', weights_only=True)
    overlap = set(sd) & set(part)
    assert not overlap, f'base와 interaction weights의 key가 겹친다: {sorted(overlap)[:5]}'
    sd.update(part)
    model.load_state_dict(sd, strict=True)  # key가 하나라도 빠지거나 남으면 여기서 실패한다
    torch.save(model.state_dict(), d / 'weights_best.pt')
    print(f'MERGED {args.dir}: base {len(sd) - len(part)} + interaction {len(part)} tensors -> {d / "weights_best.pt"}')


if __name__ == '__main__':
    main()
