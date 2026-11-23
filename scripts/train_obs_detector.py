"""E56: 화면에서 obstacle 위치·속도를 읽는 탐지기와 obstacle 궤적 예측기를 학습해 저장한다 (iawm.obs_detector).

- 탐지기: 앞쪽 --episodes개 episode의 frame을 GPU에 올려(uint8), episode 첫 frame이 아닌 row j마다
  (frame j-1, frame j) -> frame j의 obstacle (pos, vel) label로 학습한다. 뒤쪽 --val-episodes개 episode로 잰다.
- 예측기: 모든 episode의 label만으로 (pos_j, vel_j) -> (pos, vel)_{j+5d}, d = 1..6을 학습한다.
- 좌표 방향: 파란 픽셀 중심과 label을 비교해 x가 열인지 행인지 정한다(RESULT_ORIENT).

출력(마지막 줄들): RESULT_ORIENT, RESULT_DET (held-out 위치·속도 오차, 놓침·오탐 비율), RESULT_OBSFC (depth별 예측기 오차,
선형 외삽 대조). checkpoint는 $STABLEWM_HOME/checkpoints/<out>.

사용:
    python scripts/train_obs_detector.py --data r2f.h5 --out obs_det_e56.pt
"""

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT), str(ROOT / 'scripts')]

import numpy as np
import torch

from analyze_gate import episode_offsets
from iawm import obs_detector as od


def load_frames(f, lo, hi, device, chunk=2000):
    n = hi - lo
    out = torch.empty((n, od.IMG, od.IMG, 3), dtype=torch.uint8, device=device)
    for i in range(lo, hi, chunk):
        j = min(hi, i + chunk)
        out[i - lo:j - lo] = torch.from_numpy(f['pixels'][i:j]).to(device)
    return out


def pair_rows(off, ep_len, eps):
    """episode 첫 frame이 아닌 row (전역 index)."""
    return np.concatenate([off[e] + np.arange(1, ep_len[e]) for e in eps]).astype(np.int64)


@torch.no_grad()
def eval_detector(det, frames, base, rows, pos, vel, batch=256):
    pe, ve, miss, fp, n_true = [], [], 0, 0, 0
    det.eval()
    for i in range(0, len(rows), batch):
        r = rows[i:i + batch]
        P, V, M = det.detect(frames[r - 1 - base], frames[r - base])
        P, V, M = P.cpu().numpy(), V.cpu().numpy(), M.cpu().numpy()
        for b, row in enumerate(r):
            a, v_, m, fpp = od.match_error(P[b], V[b], M[b], pos[row], vel[row])
            pe += a; ve += v_; miss += m; fp += fpp; n_true += len(pos[row])
    pe, ve = np.array(pe), np.array(ve)
    return {'pos_mae': float(pe.mean()), 'pos_p90': float(np.quantile(pe, .9)), 'vel_mae': float(ve.mean()),
            'vel_p90': float(np.quantile(ve, .9)), 'miss': miss / max(n_true, 1), 'fp': fp / max(n_true, 1), 'n': n_true}


def train_forecaster(fc, pos, vel, off, ep_len, tr_eps, va_eps, device, steps, batch=4096):
    span = od.FRAMESKIP * od.N_DEPTH
    def samples(eps):
        j = np.concatenate([off[e] + np.arange(0, ep_len[e] - span) for e in eps if ep_len[e] > span]).astype(np.int64)
        K = pos.shape[1]
        jj = np.repeat(j, K); kk = np.tile(np.arange(K), len(j))
        x = np.concatenate([pos[jj, kk], vel[jj, kk]], -1)
        y = np.stack([np.concatenate([pos[jj + od.FRAMESKIP * d, kk], vel[jj + od.FRAMESKIP * d, kk]], -1)
                      for d in range(1, od.N_DEPTH + 1)], 1)
        return torch.from_numpy(x.astype(np.float32)).to(device), torch.from_numpy(y.astype(np.float32)).to(device)
    xt, yt = samples(tr_eps)
    xv, yv = samples(va_eps)
    opt = torch.optim.AdamW(fc.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    fc.train()
    for s in range(steps):
        i = torch.randint(0, len(xt), (batch,), device=device)
        pred = fc(xt[i])
        loss = (torch.nn.functional.smooth_l1_loss(fc.norm(pred), fc.norm(yt[i])))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if s % 1000 == 0:
            print(f'forecaster step {s} loss {loss.item():.5f}', flush=True)
    fc.eval()
    with torch.no_grad():
        pv = fc(xv)
        err_p = (pv[..., :2] - yv[..., :2]).norm(dim=-1).mean(0).cpu().numpy()
        err_v = (pv[..., 2:] - yv[..., 2:]).norm(dim=-1).mean(0).cpu().numpy()
        lin = xv[:, None, :2] + xv[:, None, 2:] * (od.FRAMESKIP * torch.arange(1, od.N_DEPTH + 1, device=device))[None, :, None]
        err_lin = (lin - yv[..., :2]).norm(dim=-1).mean(0).cpu().numpy()
    return err_p, err_v, err_lin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True, help='$STABLEWM_HOME/checkpoints 아래 파일 이름')
    ap.add_argument('--episodes', type=int, default=1200, help='탐지기 학습에 GPU로 올릴 앞쪽 episode 수')
    ap.add_argument('--val-episodes', type=int, default=100)
    ap.add_argument('--steps', type=int, default=12000)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--fc-steps', type=int, default=6000)
    ap.add_argument('--width', type=int, default=64)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    import h5py
    import hdf5plugin  # noqa: F401

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    home = Path(os.environ.get('STABLEWM_HOME', Path.home() / '.stable-wm'))
    t0 = time.time()
    with h5py.File(home / 'datasets' / args.data, 'r') as f:
        off, ep_len = episode_offsets(f)
        n_ep = len(ep_len)
        pos = np.nan_to_num(f['obstacle_pos'][:].astype(np.float32))
        vel = np.nan_to_num(f['obstacle_vel'][:].astype(np.float32))
        pos = pos.reshape(len(pos), -1, 2); vel = vel.reshape(len(vel), -1, 2)
        n_tr = min(args.episodes, n_ep - args.val_episodes)
        va_lo = n_ep - args.val_episodes
        # 좌표 방향 확인 (앞 300 frame)
        cr, rc, n_chk = od.blue_centroid_check(f['pixels'][:300], pos[:300])
        colrow = cr < rc
        print(f'RESULT_ORIENT colrow_err {cr:.3f} rowcol_err {rc:.3f} n {n_chk} -> xy_is_colrow {int(colrow)}', flush=True)
        assert min(cr, rc) < 2.0 and n_chk > 20, '파란 픽셀 중심이 label과 맞지 않는다 (좌표 정의 확인 필요)'
        fr_tr = load_frames(f, 0, int(off[n_tr - 1] + ep_len[n_tr - 1]), device)
        fr_va = load_frames(f, int(off[va_lo]), int(off[n_ep - 1] + ep_len[n_ep - 1]), device)
    print(f'frames: train {tuple(fr_tr.shape)} val {tuple(fr_va.shape)} ({time.time() - t0:.0f}s), obstacles per frame {pos.shape[1]}', flush=True)

    det = od.ObsDetector(width=args.width).to(device)
    det.xy_is_colrow.fill_(int(colrow))
    rows_tr = torch.from_numpy(pair_rows(off, ep_len, range(n_tr))).to(device)
    rows_va = pair_rows(off, ep_len, range(va_lo, n_ep))
    pos_t, vel_t = torch.from_numpy(pos).to(device), torch.from_numpy(vel).to(device)
    opt = torch.optim.AdamW(det.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)
    det.train()
    for s in range(args.steps):
        r = rows_tr[torch.randint(0, len(rows_tr), (args.batch,), device=device)]
        hm_l, reg = det(fr_tr[r - 1], fr_tr[r])
        hm_t, reg_t, msk = od.detector_targets(det, pos_t[r], vel_t[r])
        loss, parts = od.detector_loss(hm_l, reg, hm_t, reg_t, msk)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if s % 500 == 0 or s == args.steps - 1:
            print(f'detector step {s} loss {loss.item():.4f} {parts} ({time.time() - t0:.0f}s)', flush=True)
        if s > 0 and s % 4000 == 0:
            m = eval_detector(det, fr_va, int(off[va_lo]), rows_va[:3000], pos, vel)
            print(f'  val@{s}: {m}', flush=True)
            det.train()
    m = eval_detector(det, fr_va, int(off[va_lo]), rows_va, pos, vel)
    print(f"RESULT_DET {args.out} pos_mae {m['pos_mae']:.3f} pos_p90 {m['pos_p90']:.3f} vel_mae {m['vel_mae']:.3f} "
          f"vel_p90 {m['vel_p90']:.3f} miss {m['miss']:.4f} fp {m['fp']:.4f} n {m['n']}", flush=True)

    fc = od.ObsForecaster().to(device)
    err_p, err_v, err_lin = train_forecaster(fc, pos, vel, off, ep_len, range(0, va_lo), range(va_lo, n_ep), device, args.fc_steps)
    print('RESULT_OBSFC ' + ' '.join(f'd{d + 1} pos {err_p[d]:.2f} vel {err_v[d]:.2f} lin {err_lin[d]:.2f}' for d in range(od.N_DEPTH)), flush=True)

    out = home / 'checkpoints' / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    od.save(out, det.cpu(), fc.cpu(), {'det_kwargs': {'width': args.width}, 'xy_is_colrow': int(colrow), 'data': args.data,
                                       'det_val': m, 'fc_val_pos': err_p.tolist(), 'fc_val_vel': err_v.tolist()})
    print(f'saved {out} ({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
