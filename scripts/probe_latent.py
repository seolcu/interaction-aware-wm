"""학습된 encoder의 latent(z^c, z^u)에 obstacle과 agent 정보가 얼마나 담겼는지 probe로 확인한다.

Gate가 contact를 알아보려면 입력 latent에 obstacle 위치가 있어야 한다. 그래서 Collision TwoRoom
frame을 encoder로 보내 z^c, z^u, z(둘 다)를 뽑고, 각각에서 다음을 맞혀 본다.
- 회귀(R², test episode 기준): obstacle x·y, agent x·y, agent–obstacle 거리
- 분류(AUROC): 그 frame을 만든 step에 contact가 있었는가

Probe는 두 가지다. linear(Ridge)는 "그대로 꺼내 쓸 수 있는가", MLP는 "비선형으로라도 들어 있는가"를 본다.
Episode 단위로 앞 80%를 학습, 뒤 20%를 test로 나눈다.

사용:
    python scripts/probe_latent.py --ckpt gate_collision_n2/weights_epoch_10.pt
    python scripts/probe_latent.py --ckpt cfjepa_collision_n2/weights_epoch_10.pt --distractors 2
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT), str(ROOT / 'scripts')]

import numpy as np
import torch

from analyze_gate import IMAGENET_MEAN, IMAGENET_STD, episode_offsets, to_model_pixels


def agent_pos(f, o, L):
    for k in ('pos_agent', 'proprio'):
        if k in f:
            return f[k][o:o + L].reshape(L, -1)[:, :2].astype(np.float32)
    raise KeyError('agent position column (pos_agent / proprio) not found')


@torch.no_grad()
def extract(model, f, episodes, device, batch=256, distractors=0):
    """frame마다 latent와 정답을 모은다."""
    off, ep_len = episode_offsets(f)
    Z, OBS, AG, CON, EP = [], [], [], [], []
    for ep in episodes:
        o, L = int(off[ep]), int(ep_len[ep])
        px = f['pixels'][o:o + L]
        for i in range(0, L, batch):
            pixels = to_model_pixels(px[i:i + batch][:, None], device)  # (B, 1, 3, H, W)
            if distractors > 0:
                from iawm.distractors import DistractorConfig, overlay, sample_params
                dcfg = DistractorConfig(num=distractors, img_size=pixels.size(-1))
                params = sample_params(pixels.size(0), dcfg, device=device)
                t = torch.zeros(1, device=device)
                pixels = overlay(pixels, params, t, dcfg, IMAGENET_MEAN.view(1, 3), IMAGENET_STD.view(1, 3))
            Z.append(model.encode({'pixels': pixels})['emb'][:, 0].float().cpu().numpy())
        OBS.append(f['obstacle_pos'][o:o + L].reshape(L, -1)[:, :2].astype(np.float32))
        AG.append(agent_pos(f, o, L))
        CON.append(f['contact'][o:o + L].astype(bool))
        EP.append(np.full(L, ep))
    Z, OBS, AG, CON, EP = map(np.concatenate, (Z, OBS, AG, CON, EP))
    targets = {
        'obstacle_x': OBS[:, 0], 'obstacle_y': OBS[:, 1],
        'agent_x': AG[:, 0], 'agent_y': AG[:, 1],
        'agent_obstacle_dist': np.linalg.norm(AG - OBS, axis=1),
    }
    return Z, targets, CON, EP


def r2(y, p):
    return 1.0 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def mlp_fit_predict(Xtr, Ytr, Xte, device, classify=False, steps=3000, hidden=256, lr=1e-3, eval_every=50):
    """2-layer MLP probe. 학습 data의 10%를 떼어 validation loss가 가장 낮은 시점의 weight를 쓴다(early stopping)."""
    import copy

    torch.manual_seed(0)
    Xtr, Xte = torch.tensor(Xtr, device=device), torch.tensor(Xte, device=device)
    Ytr = torch.tensor(Ytr, device=device, dtype=torch.float32).view(len(Xtr), -1)
    perm = torch.randperm(len(Xtr), device=device)
    n_val = max(1, len(Xtr) // 10)
    vi, ti = perm[:n_val], perm[n_val:]
    net = torch.nn.Sequential(
        torch.nn.Linear(Xtr.shape[1], hidden), torch.nn.GELU(), torch.nn.Dropout(0.1),
        torch.nn.Linear(hidden, hidden), torch.nn.GELU(), torch.nn.Dropout(0.1),
        torch.nn.Linear(hidden, Ytr.shape[1]),
    ).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    loss_fn = torch.nn.BCEWithLogitsLoss() if classify else torch.nn.MSELoss()
    best, best_state = float('inf'), copy.deepcopy(net.state_dict())
    for step in range(steps):
        net.train()
        idx = ti[torch.randint(0, len(ti), (min(1024, len(ti)),), device=device)]
        opt.zero_grad()
        loss_fn(net(Xtr[idx]), Ytr[idx]).backward()
        opt.step()
        if (step + 1) % eval_every == 0:
            net.eval()
            with torch.no_grad():
                v = loss_fn(net(Xtr[vi]), Ytr[vi]).item()
            if v < best:
                best, best_state = v, copy.deepcopy(net.state_dict())
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        return net(Xte).cpu().numpy()


def probe(Z, targets, con, ep, dim_c, device):
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import roc_auc_score

    eps = np.unique(ep)
    test_eps = eps[int(0.8 * len(eps)):]
    te = np.isin(ep, test_eps)
    tr = ~te
    inputs = {'z_c': Z[:, :dim_c], 'z_u': Z[:, dim_c:], 'z (both)': Z}
    names = list(targets) + ['contact (AUROC)']
    results = {}
    for in_name, X in inputs.items():
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        X = ((X - mu) / sd).astype(np.float32)
        Y = np.stack([targets[k] for k in targets], 1)
        ym, ys = Y[tr].mean(0), Y[tr].std(0) + 1e-6
        Yn = ((Y - ym) / ys).astype(np.float32)
        lin = Ridge(alpha=1.0).fit(X[tr], Yn[tr]).predict(X[te])
        mlp = mlp_fit_predict(X[tr], Yn[tr], X[te], device)
        for j, k in enumerate(targets):
            results[(in_name, k)] = (r2(Yn[te, j], lin[:, j]), r2(Yn[te, j], mlp[:, j]))
        if con[tr].any() and con[te].any():
            lr = LogisticRegression(max_iter=2000, class_weight='balanced').fit(X[tr], con[tr])
            p_lin = lr.decision_function(X[te])
            p_mlp = mlp_fit_predict(X[tr], con[tr].astype(np.float32), X[te], device, classify=True)[:, 0]
            results[(in_name, 'contact (AUROC)')] = (roc_auc_score(con[te], p_lin), roc_auc_score(con[te], p_mlp))
    print(f'frames: train {tr.sum()}  test {te.sum()}  contact ratio {con.mean():.4f}')
    print('값: R² (회귀, 1이 완벽·0이 평균 수준) / AUROC (contact, 1이 완벽·0.5가 무작위). 각 칸은 linear | MLP')
    header = f'{"target":22s}' + ''.join(f'{n:>20s}' for n in inputs)
    print(header)
    for k in names:
        cells = ''
        for in_name in inputs:
            v = results.get((in_name, k))
            cells += f'{"-":>20s}' if v is None else f'{v[0]:>9.3f} | {v[1]:<8.3f}'.rjust(20)
        print(f'{k:22s}{cells}')
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='$STABLEWM_HOME/checkpoints 기준 경로')
    ap.add_argument('--data', default='collision_tworoom.h5')
    ap.add_argument('--episodes', type=int, default=500, help='data의 마지막 N episode를 쓴다')
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--distractors', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    import h5py
    import hdf5plugin  # noqa: F401
    import stable_worldmodel as swm

    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = swm.wm.utils.load_pretrained(args.ckpt).to(device).eval()
    dim_c = getattr(model, 'dim_c', None)
    if dim_c is None:
        raise SystemExit('z^c / z^u로 나뉜 모델(CF-JEPA 계열)만 지원한다')
    path = Path(swm.data.utils.get_cache_dir()) / 'datasets' / args.data
    with h5py.File(path, 'r') as f:
        n_ep = len(f['ep_len'])
        episodes = range(max(0, n_ep - args.episodes), n_ep)
        Z, targets, con, ep = extract(model, f, episodes, device, args.batch, args.distractors)
    print(f'ckpt={args.ckpt} data={args.data} episodes={len(episodes)} distractors={args.distractors}')
    probe(Z, targets, con, ep, dim_c, device)


if __name__ == '__main__':
    main()
