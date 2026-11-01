"""Collision TwoRoom data에서 interaction gate α와 예측 오차를 contact 라벨과 비교한다.

학습된 checkpoint를 불러 data의 마지막 N episode로 다음을 계산한다.
- α의 contact 구분 AUROC (RQ2). gate가 없는 모델(CF-JEPA)은 건너뛴다.
- contact / non-contact transition별 z^c 예측 오차 (README의 prediction error 지표).
- (E6) gate 모델이면 h_int를 끈 예측(err_c_off, CF-JEPA 부분)과 켠 예측(err_c_on, α=1)의 오차,
  상대 이득 (off - on) / off를 contact별로 비교한다. gain gate의 τ를 정하는 근거다.

Transition과 contact 라벨 맞추기:
- World.collect는 row j에 "step j-1 이후의 frame"과 그 step의 info(contact)를 저장하고,
  action은 한 칸 당겨 row j의 action이 frame j -> j+1을 만들게 저장한다.
- 학습 clip은 frame row s, s+5, s+10, s+15와 action row s..s+19다(frameskip 5).
- 모델의 position k 예측(frame s+5k -> s+5k+5)의 contact 라벨은
  contact[s+5k+1 .. s+5k+5] 중 하나라도 True인지다.

사용:
    python scripts/analyze_gate.py --ckpt gate_collision_n2/weights_epoch_10.pt
    python scripts/analyze_gate.py --ckpt cfjepa_collision_n2/weights_epoch_10.pt --out cfjepa.csv
    python scripts/analyze_gate.py --ckpt gate_diag/weights_epoch_1.pt --episodes 200
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

import numpy as np
import torch

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def episode_offsets(f):
    ep_len = f['ep_len'][:].astype(np.int64)
    if 'ep_offset' in f:
        off = f['ep_offset'][:].astype(np.int64)
    else:
        off = np.concatenate([[0], np.cumsum(ep_len)[:-1]])
    return off, ep_len


def action_stats(f):
    """학습 때 get_column_normalizer와 같게, NaN 행을 뺀 전체 action column의 평균과 표준편차."""
    a = f['action'][:].astype(np.float64)
    a = a[~np.isnan(a).any(axis=1)]
    return a.mean(0), a.std(0, ddof=1)


def iter_clips(f, episodes, num_steps=4, frameskip=5, clip_stride=5, act_mean=None, act_std=None):
    """episode마다 clip을 만든다. 반환: pixels (C, num_steps, 3, H, W) uint8, action (C, num_steps, fs*2),
    contact (C, num_steps-1) bool, episode index."""
    off, ep_len = episode_offsets(f)
    span = num_steps * frameskip
    for ep in episodes:
        o, L = int(off[ep]), int(ep_len[ep])
        if L < span:
            continue
        px = f['pixels'][o:o + L]
        act = f['action'][o:o + L].astype(np.float32)
        act = (act - act_mean) / act_std
        act = np.nan_to_num(act, nan=0.0)
        con = f['contact'][o:o + L].astype(bool)
        starts = np.arange(0, L - span + 1, clip_stride)
        idx_px = starts[:, None] + frameskip * np.arange(num_steps)[None]
        idx_act = starts[:, None] + np.arange(span)[None]
        labels = np.stack([
            np.stack([con[s + frameskip * k + 1: s + frameskip * k + frameskip + 1].any()
                      for k in range(num_steps - 1)])
            for s in starts
        ])
        yield px[idx_px], act[idx_act].reshape(len(starts), num_steps, -1), labels, ep


def to_model_pixels(px_uint8, device):
    """(B, T, H, W, 3) uint8 -> (B, T, 3, H, W) float, ImageNet 정규화 (train의 get_img_preprocessor와 같다)."""
    x = torch.from_numpy(px_uint8).to(device)
    B, T = x.shape[:2]
    x = x.permute(0, 1, 4, 2, 3).float().div_(255.0).flatten(0, 1)
    x = (x - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
    return x.view(B, T, *x.shape[1:])


@torch.no_grad()
def run(model, f, episodes, device, batch=128, distractors=0, frameskip=5, history=3):
    act_mean, act_std = action_stats(f)
    dim_c = getattr(model, 'dim_c', None)
    has_gate = hasattr(model.predictor, 'last_alpha')
    rows = {'ep': [], 'pos': [], 'contact': [], 'err_c': [], 'err_all': [], 'alpha': [],
            'err_c_off': [], 'err_c_on': []}
    buf_px, buf_act, buf_lab, buf_ep = [], [], [], []

    def flush():
        if not buf_px:
            return
        px = np.concatenate(buf_px)
        act = torch.from_numpy(np.concatenate(buf_act)).to(device)
        lab = np.concatenate(buf_lab)
        eps = np.concatenate(buf_ep)
        buf_px.clear(), buf_act.clear(), buf_lab.clear(), buf_ep.clear()
        for i in range(0, len(px), batch):
            pixels = to_model_pixels(px[i:i + batch], device)
            if distractors > 0:
                from iawm.distractors import DistractorConfig, overlay, sample_params
                dcfg = DistractorConfig(num=distractors, img_size=pixels.size(-1))
                params = sample_params(pixels.size(0), dcfg, device=device)
                t = torch.arange(pixels.size(1), device=device).float() * frameskip
                pixels = overlay(pixels, params, t, dcfg, IMAGENET_MEAN.view(1, 3), IMAGENET_STD.view(1, 3))
            out = model.encode({'pixels': pixels, 'action': act[i:i + batch]})
            emb, act_emb = out['emb'], out['act_emb']
            pred = model.predict(emb[:, :history], act_emb[:, :history])
            tgt = emb[:, 1:history + 1]
            diff = (pred - tgt).float().pow(2)
            err_all = diff.mean(-1)
            err_c = diff[..., :dim_c].mean(-1) if dim_c else err_all
            alpha = model.predictor.last_alpha.float().squeeze(-1) if has_gate else torch.full_like(err_all, float('nan'))
            base = getattr(model.predictor, 'last_pred_c_base', None)
            delta = getattr(model.predictor, 'last_delta', None)
            if has_gate and base is not None and delta is not None:
                tgt_c = tgt[..., :dim_c].float()
                err_off = (base.float() - tgt_c).pow(2).mean(-1)
                err_on = (base.float() + delta.float() - tgt_c).pow(2).mean(-1)
            else:
                err_off = err_on = torch.full_like(err_all, float('nan'))
            n, T = err_all.shape
            rows['ep'].append(np.repeat(eps[i:i + batch], T))
            rows['pos'].append(np.tile(np.arange(T), n))
            rows['contact'].append(lab[i:i + batch].reshape(-1))
            rows['err_c'].append(err_c.cpu().numpy().reshape(-1))
            rows['err_all'].append(err_all.cpu().numpy().reshape(-1))
            rows['alpha'].append(alpha.cpu().numpy().reshape(-1))
            rows['err_c_off'].append(err_off.cpu().numpy().reshape(-1))
            rows['err_c_on'].append(err_on.cpu().numpy().reshape(-1))

    n_buf = 0
    for px, act, lab, ep in iter_clips(f, episodes, history + 1, frameskip, frameskip, act_mean, act_std):
        buf_px.append(px), buf_act.append(act), buf_lab.append(lab), buf_ep.append(np.full(len(px), ep))
        n_buf += len(px)
        if n_buf >= 4 * batch:
            flush()
            n_buf = 0
    flush()
    return {k: np.concatenate(v) for k, v in rows.items()}, has_gate


def report(r, has_gate):
    from sklearn.metrics import roc_auc_score

    c = r['contact']
    print(f'transitions: {len(c)}  contact ratio: {c.mean():.4f}')
    for name in ('err_c', 'err_all'):
        e = r[name]
        print(f'{name:8s} all={e.mean():.5f}  non-contact={e[~c].mean():.5f}  contact={e[c].mean():.5f}  '
              f'ratio(contact/non)={e[c].mean() / e[~c].mean():.2f}')
    if has_gate:
        a = r['alpha']
        q = np.quantile(a, [0.0, 0.1, 0.5, 0.9, 1.0])
        print(f'alpha    quantiles(0,10,50,90,100%)=' + ' '.join(f'{v:.4g}' for v in q))
        print(f'alpha    non-contact mean={a[~c].mean():.4g}  contact mean={a[c].mean():.4g}')
        if c.any() and (~c).any():
            print(f'AUROC(alpha -> contact) = {roc_auc_score(c, a):.4f}')
            for p in np.unique(r['pos']):
                m = r['pos'] == p
                if c[m].any() and (~c[m]).any():
                    print(f'  position {p}: AUROC = {roc_auc_score(c[m], a[m]):.4f}')
    off, on = r['err_c_off'], r['err_c_on']
    if has_gate and np.isfinite(off).all() and np.isfinite(on).all():
        g = (off - on) / np.maximum(off, 1e-8)
        print('--- h_int 이득 (α=1로 켰을 때 z^c 오차가 얼마나 줄었나) ---')
        for name, m in (('non-contact', ~c), ('contact', c)):
            q = np.quantile(g[m], [0.1, 0.5, 0.9, 0.99])
            print(f'{name:12s} err_off={off[m].mean():.5f} err_on={on[m].mean():.5f} '
                  f'rel_gain mean={g[m].mean():.4f} q10/50/90/99=' + '/'.join(f'{v:.3f}' for v in q))
        if c.any() and (~c).any():
            print(f'AUROC(rel_gain -> contact) = {roc_auc_score(c, g):.4f}   '
                  f'AUROC(abs_gain -> contact) = {roc_auc_score(c, off - on):.4f}')
        for tau in (0.02, 0.05, 0.1, 0.2, 0.3):
            print(f'  τ={tau:<4}: 목표=1 비율  all={(g > tau).mean():.4f}  non-contact={(g[~c] > tau).mean():.4f}  '
                  f'contact={(g[c] > tau).mean():.4f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='$STABLEWM_HOME/checkpoints 기준 경로 (예: gate_collision_n2/weights_epoch_10.pt)')
    ap.add_argument('--data', default='collision_tworoom.h5')
    ap.add_argument('--episodes', type=int, default=500, help='data의 마지막 N episode를 쓴다')
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--distractors', type=int, default=0, help='평가 이미지에 덧그릴 사각형 수 (학습과 같은 방식)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None, help='transition별 결과를 저장할 csv 경로')
    args = ap.parse_args()

    import h5py
    import hdf5plugin  # noqa: F401  (blosc 압축 해제)
    import stable_worldmodel as swm

    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = swm.wm.utils.load_pretrained(args.ckpt).to(device).eval()
    path = Path(swm.data.utils.get_cache_dir()) / 'datasets' / args.data
    with h5py.File(path, 'r') as f:
        n_ep = len(f['ep_len'])
        episodes = range(max(0, n_ep - args.episodes), n_ep)
        r, has_gate = run(model, f, episodes, device, batch=args.batch, distractors=args.distractors)
    print(f'ckpt={args.ckpt} data={args.data} episodes={len(episodes)} distractors={args.distractors}')
    report(r, has_gate)
    if args.out:
        import pandas as pd
        pd.DataFrame(r).to_csv(args.out, index=False)
        print(f'saved {args.out}')


if __name__ == '__main__':
    main()
