"""여러 step 앞 z^c 예측 오차를 모델끼리 비교한다 (E25 진단).

history 3 frame에서 시작해 예측을 입력으로 되먹이며 k step 앞까지 간다. 모델마다 latent 크기가 다르므로
오차를 그대로 비교하지 않고, 마지막 관측 z^c를 그대로 쓴 예측(persistence)의 오차로 나눈 비율을 쓴다.
낮을수록 좋다. contact 열은 1~k step 사이에 충돌이 있는 clip만 모은 값이다.

사용:
    python scripts/multistep_error.py --ckpts e22_cf/weights_epoch_4.pt,e25_cf/weights_epoch_4.pt
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT), str(ROOT / 'scripts')]

import numpy as np
import torch

from analyze_gate import action_stats, iter_clips, to_model_pixels, IMAGENET_MEAN, IMAGENET_STD


@torch.no_grad()
def errors(model, f, episodes, device, steps, history=3, frameskip=5, distractors=2, batch=128):
    from iawm.distractors import DistractorConfig, overlay, sample_params
    act_mean, act_std = action_stats(f)
    d_c = model.dim_c
    pred, pers, labs = [], [], []
    for px, act, lab, _, _ in iter_clips(f, episodes, history + steps, frameskip, frameskip, act_mean, act_std):
        act = torch.from_numpy(act).to(device)
        for i in range(0, len(px), batch):
            pixels = to_model_pixels(px[i:i + batch], device)
            if distractors > 0:
                dcfg = DistractorConfig(num=distractors, img_size=pixels.size(-1))
                params = sample_params(pixels.size(0), dcfg, device=device)
                t = torch.arange(pixels.size(1), device=device).float() * frameskip
                pixels = overlay(pixels, params, t, dcfg, IMAGENET_MEAN.view(1, 3), IMAGENET_STD.view(1, 3))
            o = model.encode({'pixels': pixels, 'action': act[i:i + batch]})
            emb, act_emb = o['emb'], o['act_emb']
            window = emb[:, :history]
            e_pred, e_pers = [], []
            for k in range(steps):
                nxt = model.predict(window, act_emb[:, k:k + history])[:, -1:]
                tgt = emb[:, history + k, :d_c]
                e_pred.append((nxt[:, 0, :d_c] - tgt).pow(2).mean(-1).float().cpu())
                e_pers.append((emb[:, history - 1, :d_c] - tgt).pow(2).mean(-1).float().cpu())
                window = torch.cat([window[:, 1:], nxt], dim=1)
            pred.append(torch.stack(e_pred, 1)); pers.append(torch.stack(e_pers, 1))
            labs.append(torch.from_numpy(lab[i:i + batch, history - 1:]))
    return torch.cat(pred), torch.cat(pers), torch.cat(labs).bool()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpts', required=True)
    ap.add_argument('--data', default='collision_tworoom_contact.h5')
    ap.add_argument('--episodes', type=int, default=500)
    ap.add_argument('--steps', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    import h5py
    import hdf5plugin  # noqa: F401
    import stable_worldmodel as swm

    path = Path(swm.data.utils.get_cache_dir()) / 'datasets' / args.data
    print(f'{"ckpt":<30}{"k":>3}{"clips":>7}{"all":>8}{"no-contact":>12}{"contact":>9}')
    with h5py.File(path, 'r') as f:
        n_ep = len(f['ep_len'])
        for ckpt in args.ckpts.split(','):
            torch.manual_seed(args.seed); np.random.seed(args.seed)
            model = swm.wm.utils.load_pretrained(ckpt).to('cuda').eval()
            pred, pers, lab = errors(model, f, range(n_ep - args.episodes, n_ep), 'cuda', args.steps)
            for k in range(args.steps):
                c = lab[:, :k + 1].any(1)
                ratio = lambda m: (pred[m, k].mean() / pers[m, k].mean()).item()
                print(f'{ckpt:<30}{k + 1:>3}{len(pred):>7}{ratio(torch.ones_like(c)):>8.3f}{ratio(~c):>12.3f}{ratio(c):>9.3f}', flush=True)
            del model


if __name__ == '__main__':
    main()
