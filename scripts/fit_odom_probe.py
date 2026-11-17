"""위치 label 없이 action만으로 z^c -> agent 위치 probe를 학습한다 (E43).

E40/E42의 위치 cost(`cost=agent_px`)는 정답 agent 위치로 학습한 probe를 쓴다. 여기서는 위치 label 대신 data에 이미 있는
action을 쓴다. TwoRoom에서 agent는 한 step에 `clip(a, -1, 1) * speed`만큼 움직이므로(벽과 obstacle에 막히거나 밀리지
않으면), 같은 episode의 두 frame t < t+k 사이 이동은 action 누적합으로 정해진다.

    f(z_{t+k}) - f(z_t) ≈ speed * Σ_{i=t}^{t+k-1} clip(a_i, -1, 1)

를 맞히도록 MLP f를 학습한다. 벽에 막히거나 obstacle에 밀린 구간은 이 식이 틀리므로 Huber loss로 영향을 줄인다(contact
label도 쓰지 않는다). f는 상수만큼의 평행 이동이 정해지지 않지만, planning cost는 예측과 goal의 차이만 쓰므로 상관없다.
결과 probe는 `iawm.pos_probe` 형식(출력 2차원)으로 저장해 `+cost=agent_px +pos_probe=<파일>`에 그대로 쓴다.

평가에만 정답 위치를 쓴다(학습에는 쓰지 않는다).
- disp: 같은 episode의 임의의 두 frame에 대해 f 차이와 실제 위치 차이의 거리(px)
- abs: held-out frame에서 f에 평행 이동 하나만 맞춘 뒤(학습 frame에서 평균 차이로 정한다) 실제 위치와의 거리(px)
- affine: 참고로 f -> 위치 affine 변환을 학습 frame에서 맞춘 뒤의 held-out 거리(px). abs와 차이가 크면 축척이나
  회전이 어긋난 것이다.

사용:
    python scripts/fit_odom_probe.py --base e32_cf/weights_epoch_10.pt --data r1p.h5 --out odom_probe_e32_cf.pt
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT), str(ROOT / 'scripts')]

import numpy as np
import torch


def episode_bounds(f):
    ep_len = f['ep_len'][:].astype(np.int64)
    off = np.concatenate([[0], np.cumsum(ep_len)[:-1]])
    return off, ep_len


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True, help='$STABLEWM_HOME/checkpoints 아래 파일 이름')
    ap.add_argument('--speed', type=float, default=5.0, help='agent 속도(px/step). TwoRoom 기본값 5')
    ap.add_argument('--max-k', type=int, default=10, help='쌍의 최대 간격(step)')
    ap.add_argument('--shift', type=int, default=0, help='action row i가 frame i -> i+1 이동이면 0')
    ap.add_argument('--steps', type=int, default=8000)
    ap.add_argument('--huber', type=float, default=2.0, help='Huber loss의 전환점(px)')
    ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    import h5py
    import hdf5plugin  # noqa: F401
    import stable_worldmodel as swm

    from iawm import pos_probe
    from multistep_px import positions
    from train_int_frozen import frame_latents

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    home = Path(swm.data.utils.get_cache_dir())
    model = swm.wm.utils.load_pretrained(args.base).to(device).eval()
    tag = args.base.replace('/', '_').replace('.pt', '')
    with h5py.File(home / 'datasets' / args.data, 'r') as f:
        z = frame_latents(model, f, device, home / 'latent_cache' / f'{tag}__{Path(args.data).stem}.npy')
        act = np.clip(np.nan_to_num(f['action'][:].reshape(len(z), -1)[:, :2].astype(np.float32)), -1, 1)
        true = positions(f)[:, :2]  # 평가에만 쓴다
        off, ep_len = episode_bounds(f)
    z = torch.from_numpy(z[:, :model.dim_c].astype(np.float32)).to(device)
    n = len(z)

    # episode를 학습(95%)과 held-out(5%)으로 나눈다
    rng = np.random.default_rng(args.seed)
    eps = rng.permutation(len(off))
    n_te = max(20, len(off) // 20)
    te_eps, tr_eps = eps[:n_te], eps[n_te:]

    # 누적 action: row o+m (episode 시작 o)에 대해 cum[o+m] = Σ_{i<m} a_{o+i}. 같은 episode의 t < t2 사이 이동은
    # speed * (cum[t2] - cum[t]) = speed * Σ_{i=t}^{t2-1} a_i 다 (shift 반영).
    cum = np.zeros((n, 2), np.float32)
    ep_id = np.zeros(n, np.int64)
    for e, (o, L) in enumerate(zip(off, ep_len)):
        a = act[o:o + L]
        if args.shift:
            a = np.concatenate([np.zeros((args.shift, 2), np.float32), a[:-args.shift]])
        cum[o:o + L] = np.concatenate([np.zeros((1, 2), np.float32), np.cumsum(a, 0)[:-1]])
        ep_id[o:o + L] = e
    cum_t = torch.from_numpy(cum).to(device)

    def rows_of(ep_list):
        return np.concatenate([np.arange(off[e], off[e] + ep_len[e]) for e in ep_list])

    tr_rows = torch.from_numpy(rows_of(tr_eps)).to(device)
    ep_end = torch.from_numpy((off + ep_len)[ep_id]).to(device)  # row별 episode 끝(제외)

    torch.manual_seed(args.seed)
    probe = pos_probe.PosProbe(z.shape[1], args.hidden, 2).to(device)
    probe.xm.copy_(z[tr_rows].mean(0)), probe.xs.copy_(z[tr_rows].std(0) + 1e-6)
    probe.ym.zero_(), probe.ys.fill_(args.speed * 10)
    opt = torch.optim.AdamW(probe.net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    for it in range(args.steps):
        t = tr_rows[torch.randint(0, len(tr_rows), (4096,), device=device)]
        k = torch.randint(1, args.max_k + 1, (len(t),), device=device)
        t2 = torch.minimum(t + k, ep_end[t] - 1)
        ok = t2 > t
        t, t2 = t[ok], t2[ok]
        target = args.speed * (cum_t[t2] - cum_t[t])
        pred = probe(z[t2]) - probe(z[t])
        loss = torch.nn.functional.huber_loss(pred, target, delta=args.huber)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if it % 2000 == 0:
            print(f'step {it} loss {loss.item():.3f}', flush=True)
    probe.eval().requires_grad_(False)

    # ---- 평가 (정답 위치는 여기서만 쓴다) ----
    with torch.no_grad():
        F = torch.cat([probe(z[i:i + 20000]) for i in range(0, n, 20000)]).cpu().numpy()
    tr_r, te_r = rows_of(tr_eps), rows_of(te_eps)
    shift = (true[tr_r] - F[tr_r]).mean(0)
    abs_err = np.linalg.norm(F[te_r] + shift - true[te_r], axis=1).mean()
    X = np.concatenate([F, np.ones((n, 1), np.float32)], 1)
    W, *_ = np.linalg.lstsq(X[tr_r], true[tr_r], rcond=None)
    aff_err = np.linalg.norm(X[te_r] @ W - true[te_r], axis=1).mean()
    # 같은 episode의 임의의 두 frame
    d = []
    for e in te_eps:
        r = np.arange(off[e], off[e] + ep_len[e])
        i, j = rng.choice(r, 50), rng.choice(r, 50)
        d.append(np.linalg.norm((F[j] - F[i]) - (true[j] - true[i]), axis=1))
    disp_err = np.concatenate(d).mean()
    spread = np.linalg.norm(true[te_r] - true[te_r].mean(0), axis=1).mean()
    print(f'held-out: disp {disp_err:.2f}px abs {abs_err:.2f}px affine {aff_err:.2f}px '
          f'(affine scale {np.linalg.norm(W[:2], axis=0).round(3).tolist()}, position spread {spread:.1f}px)')
    pos_probe.save(probe, home / 'checkpoints' / args.out)
    print(f'RESULT_ODOM {args.out} frames {n} disp_px {disp_err:.3f} abs_px {abs_err:.3f} affine_px {aff_err:.3f}')


if __name__ == '__main__':
    main()
