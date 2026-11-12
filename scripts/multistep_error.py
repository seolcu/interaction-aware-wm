"""여러 step 앞 z^c 예측 오차를 모델끼리 비교한다 (E25 진단).

history 3 frame에서 시작해 예측을 입력으로 되먹이며 k step 앞까지 간다. 모델마다 latent 크기가 다르므로
오차를 그대로 비교하지 않고, 마지막 관측 z^c를 그대로 쓴 예측(persistence)의 오차로 나눈 비율을 쓴다.
낮을수록 좋다. contact 열은 1~k step 사이에 충돌이 있는 clip만 모은 값이다.

E27에서 더한 것: gate 모델이면 상상 rollout의 k번째 step에서 gate가 낸 α를 기록하고, 그 step의 실제 contact와의
AUROC(`imag_auroc`)를 낸다. 예측한 z^u로도 gate가 미래 충돌에 열리는지 보는 값이다. `--gate-modes learned,off`로
같은 모델을 gate를 끄고도 잰다. `--distractors 0`은 distractor 없이 학습한 모델용이다.

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
    """반환: 예측 오차, persistence 오차 (N, steps), contact 라벨 (N, steps), 상상 α (N, steps) 또는 None."""
    from iawm.distractors import DistractorConfig, overlay, sample_params
    act_mean, act_std = action_stats(f)
    d_c = model.dim_c
    pred, pers, labs, alphas = [], [], [], []
    gated = hasattr(model.predictor, 'last_alpha')
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
            e_pred, e_pers, e_alpha = [], [], []
            for k in range(steps):
                nxt = model.predict(window, act_emb[:, k:k + history])[:, -1:]
                if gated:
                    e_alpha.append(model.predictor.last_alpha[:, -1, 0].float().cpu())
                tgt = emb[:, history + k, :d_c]
                e_pred.append((nxt[:, 0, :d_c] - tgt).pow(2).mean(-1).float().cpu())
                e_pers.append((emb[:, history - 1, :d_c] - tgt).pow(2).mean(-1).float().cpu())
                window = torch.cat([window[:, 1:], nxt], dim=1)
            pred.append(torch.stack(e_pred, 1)); pers.append(torch.stack(e_pers, 1))
            if gated:
                alphas.append(torch.stack(e_alpha, 1))
            labs.append(torch.from_numpy(lab[i:i + batch, history - 1:]))
    return torch.cat(pred), torch.cat(pers), torch.cat(labs).bool(), (torch.cat(alphas) if gated else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpts', required=True)
    ap.add_argument('--data', default='collision_tworoom_contact.h5')
    ap.add_argument('--episodes', type=int, default=500)
    ap.add_argument('--steps', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--distractors', type=int, default=2)
    ap.add_argument('--gate-modes', default='learned', help='gate 모델에 적용할 gate_mode 목록 (예: learned,off)')
    args = ap.parse_args()

    import h5py
    import hdf5plugin  # noqa: F401
    import stable_worldmodel as swm

    path = Path(swm.data.utils.get_cache_dir()) / 'datasets' / args.data
    from sklearn.metrics import roc_auc_score
    print(f'{"ckpt":<36}{"k":>3}{"clips":>7}{"all":>8}{"no-contact":>12}{"contact":>9}{"imag_auroc":>12}')
    with h5py.File(path, 'r') as f:
        n_ep = len(f['ep_len'])
        for ckpt in args.ckpts.split(','):
            model = swm.wm.utils.load_pretrained(ckpt).to('cuda').eval()
            modes = args.gate_modes.split(',') if hasattr(model.predictor, 'gate_mode') else ['-']
            for mode in modes:
                if mode != '-':
                    model.predictor.gate_mode = mode
                torch.manual_seed(args.seed); np.random.seed(args.seed)  # gate mode끼리 같은 distractor
                pred, pers, lab, alpha = errors(model, f, range(n_ep - args.episodes, n_ep), 'cuda', args.steps,
                                                distractors=args.distractors)
                name = ckpt if mode == '-' else f'{ckpt}[{mode}]'
                for k in range(args.steps):
                    c = lab[:, :k + 1].any(1)
                    ratio = lambda m: (pred[m, k].mean() / pers[m, k].mean()).item()
                    au = float('nan')
                    if alpha is not None and mode == 'learned' and 0 < lab[:, k].sum() < len(lab):
                        au = roc_auc_score(lab[:, k].numpy(), alpha[:, k].numpy())
                    print(f'{name:<36}{k + 1:>3}{len(pred):>7}{ratio(torch.ones_like(c)):>8.3f}{ratio(~c):>12.3f}'
                          f'{ratio(c):>9.3f}{au:>12.3f}', flush=True)
            del model


if __name__ == '__main__':
    main()
