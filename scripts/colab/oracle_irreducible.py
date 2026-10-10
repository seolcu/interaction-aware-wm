"""C. Oracle 실험: z^u를 보지 않는 predictor에는 줄일 수 없는 collision 오차가 남는가.

Pixel과 encoder 없이 ground-truth state로 비교한다. 입력에 obstacle 정보가 있느냐 없느냐만 다르다.
  g      : (agent pos, action)                     -> Δ agent pos   # CF-JEPA의 g(z^c, a)에 해당
  g + u  : (agent pos, obstacle pos/vel, action)   -> Δ agent pos   # u -> c 경로가 있는 predictor
g는 hidden 256과 1024 두 크기로 학습한다. 크기를 키워도 contact 오차가 줄지 않는지 보기 위해서다.
대조군으로 obstacle이 없는 환경(num_obstacles=0)에서도 g를 학습한다.

사용: PYTHONPATH=/content/iawm python oracle_irreducible.py --out /content/outputs/oracle
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from demo_collision import scripted_action  # noqa: E402

from envs.collision_tworoom import CollisionTwoRoomEnv  # noqa: E402

SCALE = 224.0  # 좌표를 [0, 1] 근처로 정규화


def collect(num_obstacles, episodes, max_steps, seed0, noise):
    env = CollisionTwoRoomEnv(num_obstacles=num_obstacles)
    rows = []
    for ep in range(episodes):
        env.reset(seed=seed0 + ep)
        rng = np.random.default_rng(seed0 + ep)
        for _ in range(max_steps):
            a = scripted_action(env, rng, noise=noise)
            pos = env.agent_position.numpy().copy()
            if num_obstacles:
                opos = env.obstacle_pos[0].numpy().copy()
                ovel = env.obstacle_vel[0].numpy().copy()
            else:
                opos = ovel = np.zeros(2, dtype=np.float32)
            _, _, term, _, info = env.step(a)
            nxt = env.agent_position.numpy()
            rows.append(np.concatenate([pos, opos, ovel, a, nxt - pos, [info['contact']]]))
            if term:
                break
    d = np.asarray(rows, dtype=np.float32)
    return {
        'pos': d[:, 0:2] / SCALE,
        'obs': np.concatenate([d[:, 2:4] / SCALE, d[:, 4:6] / 10.0], 1),
        'act': d[:, 6:8],
        'delta': d[:, 8:10],  # px 단위 그대로 예측한다
        'contact': d[:, 10].astype(bool),
    }


def mlp(din, hidden):
    return nn.Sequential(
        nn.Linear(din, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, 2),
    )


def inputs(data, use_u):
    parts = [data['pos'], data['act']] + ([data['obs']] if use_u else [])
    return torch.as_tensor(np.concatenate(parts, 1))


def train_eval(train, test, use_u, hidden, epochs, seed, device):
    torch.manual_seed(seed)
    x, y = inputs(train, use_u).to(device), torch.as_tensor(train['delta']).to(device)
    xt, yt = inputs(test, use_u).to(device), torch.as_tensor(test['delta']).to(device)
    model = mlp(x.shape[1], hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n, bs = len(x), 512
    curve = []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = ((model(x[idx]) - y[idx]) ** 2).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        if (ep + 1) % max(1, epochs // 5) == 0 or ep == epochs - 1:
            curve.append((ep + 1, split_rmse(model, xt, yt, test['contact'])))
    return split_rmse(model, xt, yt, test['contact']), curve


@torch.no_grad()
def split_rmse(model, xt, yt, contact):
    model.eval()
    se = ((model(xt) - yt) ** 2).sum(1).cpu().numpy()  # px^2
    c = contact
    return {
        'all': float(np.sqrt(se.mean())),
        'non_contact': float(np.sqrt(se[~c].mean())),
        'contact': float(np.sqrt(se[c].mean())) if c.any() else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='outputs/oracle')
    ap.add_argument('--episodes', type=int, default=3000)
    ap.add_argument('--max-steps', type=int, default=100)
    ap.add_argument('--noise', type=float, default=0.5)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--seeds', type=int, default=3)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    t0 = time.time()
    n_test = args.episodes // 5
    data = {
        'collision': (collect(1, args.episodes, args.max_steps, 0, args.noise),
                      collect(1, n_test, args.max_steps, 10**6, args.noise)),
        'no_obstacle': (collect(0, args.episodes, args.max_steps, 0, args.noise),
                        collect(0, n_test, args.max_steps, 10**6, args.noise)),
    }
    for k, (tr, te) in data.items():
        print(f'[data] {k}: train {len(tr["delta"])}, test {len(te["delta"])}, '
              f'test contact {te["contact"].mean():.3f}  ({time.time() - t0:.0f}s)')

    configs = [
        ('no_obstacle', 'g (h=256)', False, 256),
        ('collision', 'g (h=256)', False, 256),
        ('collision', 'g (h=1024)', False, 1024),
        ('collision', 'g + u (h=256)', True, 256),
    ]
    results = []
    for env_name, name, use_u, hidden in configs:
        tr, te = data[env_name]
        runs = []
        for s in range(args.seeds):
            r, curve = train_eval(tr, te, use_u, hidden, args.epochs, s, device)
            runs.append(r)
        agg = {}
        for key in ('all', 'non_contact', 'contact'):
            vals = [r[key] for r in runs if r[key] is not None]
            agg[key] = (float(np.mean(vals)), float(np.std(vals))) if vals else None
        results.append({'env': env_name, 'model': name, 'rmse_px': agg, 'curve_seed_last': curve})
        print(f'[{env_name:11s}] {name:14s} ' + '  '.join(
            f'{k}={v[0]:.2f}±{v[1]:.2f}' if v else f'{k}=-' for k, v in agg.items()) + f'  ({time.time() - t0:.0f}s)')

    meta = vars(args) | {'device': device, 'unit': 'RMSE of next-step agent displacement, pixels'}
    (out / 'oracle_results.json').write_text(json.dumps({'meta': meta, 'results': results}, indent=2))
    print('saved', out / 'oracle_results.json')


if __name__ == '__main__':
    main()
