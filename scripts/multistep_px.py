"""상상 rollout의 z^c 예측을 px 단위 위치 오차로 읽는다 (E39 추가 진단).

`multistep_error.py`는 z^c 전체의 오차를 잰다. 그런데 E38에서 CF-JEPA의 z^c에는 agent뿐 아니라 obstacle 위치도
들어 있었다(1.8px로 읽힌다). 그래서 z^c 오차가 줄어도 그것이 "agent가 어디로 밀리는지"를 더 잘 맞힌 것인지,
"obstacle이 어디로 가는지"를 더 잘 맞힌 것인지 알 수 없다. planning 성공을 정하는 것은 agent 위치다.

여기서는 실제 frame의 z^c에서 agent 위치와 obstacle 위치(px)를 읽는 probe(MLP)를 먼저 학습하고, 같은 probe를
상상 rollout이 낸 z^c에 적용한다. 모델들은 같은 encoder(고정한 base)를 쓰므로 probe 하나를 같이 쓴다.
- agent / obstacle: k block 앞 예측에서 읽은 위치와 실제 위치의 거리(px) 평균
- floor: 실제 frame의 z^c에서 읽었을 때의 오차(probe 자체의 한계)
contact는 1~k block 사이에 충돌이 있는 clip이다(multistep_error.py와 같다).

사용:
    python scripts/multistep_px.py --ckpts e32_cf/weights_epoch_10.pt,e39_obsbig/weights_best.pt \
        --data r1yt.h5 --probe-data r1y.h5 --probe-cache e32_cf_weights_epoch_10__r1y.npy --gate-modes learned,always
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT), str(ROOT / 'scripts')]

import numpy as np
import torch

from analyze_gate import action_stats, iter_clips, load_oracle, to_model_pixels


def positions(f):
    """row별 [agent x, y, obstacle x, y] (px). obstacle은 첫 번째 것."""
    n = len(f['obstacle_pos'])
    key = 'pos_agent' if 'pos_agent' in f else 'proprio'
    ag = f[key][:].reshape(n, -1)[:, :2].astype(np.float32)
    ob = f['obstacle_pos'][:].reshape(n, -1)[:, :2].astype(np.float32)
    return np.nan_to_num(np.concatenate([ag, ob], 1))


def fit_probe(X, Y, device, steps=6000, hidden=256, seed=0):
    """z^c -> [agent xy, obstacle xy] MLP. 마지막 5%는 학습에 쓰지 않고 오차를 잰다. 반환: 함수, (agent, obstacle) px 오차."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(X), generator=g)
    n_te = max(1000, len(X) // 20)
    te, tr = idx[:n_te], idx[n_te:]
    X, Y = torch.from_numpy(X).to(device), torch.from_numpy(Y).to(device)
    xm, xs = X[tr].mean(0), X[tr].std(0) + 1e-6
    ym, ys = Y[tr].mean(0), Y[tr].std(0) + 1e-6
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.GELU(), torch.nn.Linear(hidden, hidden),
                              torch.nn.GELU(), torch.nn.Linear(hidden, Y.shape[1])).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    tr = tr.to(device)
    for _ in range(steps):
        b = tr[torch.randint(0, len(tr), (2048,), device=device)]
        loss = (net((X[b] - xm) / xs) - (Y[b] - ym) / ys).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    net.eval()

    @torch.no_grad()
    def probe(z):
        return net((z.float() - xm) / xs) * ys + ym

    p = probe(X[te.to(device)])
    d = (p - Y[te.to(device)])
    return probe, (d[:, :2].norm(dim=-1).mean().item(), d[:, 2:].norm(dim=-1).mean().item())


@torch.no_grad()
def rollout_px(model, f, episodes, device, steps, probe, pos, history=3, frameskip=5, batch=128):
    """반환: agent 오차, obstacle 오차, floor(agent) (N, steps) px, contact 라벨 (N, steps)."""
    act_mean, act_std = action_stats(f)
    d_c = model.dim_c
    oracle = load_oracle(model, f, frameskip)
    D = 0 if oracle is None else oracle.shape[1]
    big = pos if oracle is None else np.concatenate([oracle, pos], 1)  # iter_clips가 clip frame별로 잘라 준다
    out = {'ag': [], 'ob': [], 'floor': [], 'lab': []}
    for px, act, lab, _, orc in iter_clips(f, episodes, history + steps, frameskip, frameskip, act_mean, act_std, big):
        act, orc = torch.from_numpy(act).to(device), torch.from_numpy(orc).to(device)
        for i in range(0, len(px), batch):
            o = model.encode({'pixels': to_model_pixels(px[i:i + batch], device), 'action': act[i:i + batch]})
            emb, act_emb = o['emb'], o['act_emb']
            window = emb[:, :history]
            true = orc[i:i + batch, :, D:]  # (b, T, 4)
            e_ag, e_ob, e_fl = [], [], []
            for k in range(steps):
                if D > 0:
                    model.predictor.oracle = orc[i:i + batch, k:k + history, :D]
                nxt = model.predict(window, act_emb[:, k:k + history])[:, -1:]
                t = true[:, history + k]
                p = probe(nxt[:, 0, :d_c])
                e_ag.append((p[:, :2] - t[:, :2]).norm(dim=-1).cpu())
                e_ob.append((p[:, 2:] - t[:, 2:]).norm(dim=-1).cpu())
                e_fl.append((probe(emb[:, history + k, :d_c])[:, :2] - t[:, :2]).norm(dim=-1).cpu())
                window = torch.cat([window[:, 1:], nxt], dim=1)
            out['ag'].append(torch.stack(e_ag, 1)), out['ob'].append(torch.stack(e_ob, 1))
            out['floor'].append(torch.stack(e_fl, 1))
            out['lab'].append(torch.from_numpy(lab[i:i + batch, history - 1:]))
    return {k: torch.cat(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpts', required=True)
    ap.add_argument('--data', required=True, help='rollout을 재는 data (test)')
    ap.add_argument('--probe-data', required=True, help='probe를 학습할 data (h5)')
    ap.add_argument('--probe-cache', required=True, help='probe-data의 frame latent (latent_cache 아래 npy, train_int_frozen.py가 만든 것)')
    ap.add_argument('--episodes', type=int, default=1500)
    ap.add_argument('--steps', type=int, default=5)
    ap.add_argument('--gate-modes', default='learned')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    import h5py
    import hdf5plugin  # noqa: F401
    import stable_worldmodel as swm

    device = 'cuda'
    home = Path(swm.data.utils.get_cache_dir())
    z = np.load(home / 'latent_cache' / args.probe_cache)
    with h5py.File(home / 'datasets' / args.probe_data, 'r') as f:
        ppos = positions(f)
    assert len(z) == len(ppos), (z.shape, ppos.shape)
    models = {c: swm.wm.utils.load_pretrained(c).to(device).eval() for c in args.ckpts.split(',')}
    d_c = next(iter(models.values())).dim_c
    rng = np.random.default_rng(args.seed)
    sub = rng.choice(len(z), size=min(400000, len(z)), replace=False)
    probe, (pa, po) = fit_probe(z[sub, :d_c].astype(np.float32), ppos[sub], device, seed=args.seed)
    print(f'probe (z^c -> px) held-out error: agent {pa:.2f} px, obstacle {po:.2f} px', flush=True)
    print(f'{"ckpt":<36}{"k":>3}{"clips":>7} | agent px: {"no-contact":>10}{"contact":>9}{"floor":>7} | obstacle px: {"no-contact":>10}{"contact":>9}')
    with h5py.File(home / 'datasets' / args.data, 'r') as f:
        n_ep = len(f['ep_len'])
        pos = positions(f)
        for ckpt, model in models.items():
            modes = args.gate_modes.split(',') if hasattr(model.predictor, 'gate_mode') else ['-']
            for mode in modes:
                if mode != '-':
                    model.predictor.gate_mode = mode
                r = rollout_px(model, f, range(max(0, n_ep - args.episodes), n_ep), device, args.steps, probe, pos)
                name = ckpt if mode == '-' else f'{ckpt}[{mode}]'
                for k in range(args.steps):
                    c = r['lab'][:, :k + 1].any(1)
                    m = lambda key, mask: r[key][mask, k].mean().item()  # noqa: E731
                    print(f'{name:<36}{k + 1:>3}{len(c):>7} | {"":9}{m("ag", ~c):>10.2f}{m("ag", c):>9.2f}{m("floor", c):>7.2f} | '
                          f'{"":12}{m("ob", ~c):>10.2f}{m("ob", c):>9.2f}', flush=True)
                    print(f'RESULT_PX {name} k {k + 1} agent_nc {m("ag", ~c):.3f} agent_c {m("ag", c):.3f} floor_c {m("floor", c):.3f} '
                          f'obs_nc {m("ob", ~c):.3f} obs_c {m("ob", c):.3f}', flush=True)
    print('PX DONE')


if __name__ == '__main__':
    main()
