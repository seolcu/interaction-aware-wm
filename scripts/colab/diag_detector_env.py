"""E56 진단: env에서 직접 화면을 만들어 탐지기의 놓침이 어떤 상황에서 생기는지 본다.

Collision env를 만들고 무작위 action으로 굴리며, 매 step (앞 화면, 지금 화면)으로 탐지한다. 정답 obstacle마다
놓쳤는지와 그때의 상황(다른 obstacle과의 거리, agent와의 거리, 방 경계와의 거리)을 모아 놓침 비율을 상황별로 낸다.

사용: python scripts/colab/diag_detector_env.py --detector smk_det.pt --num-obstacles 2 --episodes 20
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

import gymnasium as gym
import numpy as np
import torch

import envs.collision_tworoom  # noqa: F401  (env 등록)
from iawm import obs_detector as od
from iawm import obs_oracle as oo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--detector', required=True)
    ap.add_argument('--num-obstacles', type=int, default=2)
    ap.add_argument('--episodes', type=int, default=20)
    ap.add_argument('--steps', type=int, default=60)
    ap.add_argument('--thr', type=float, default=0.3)
    ap.add_argument('--min-sep', type=float, default=0.0, help='E58: env의 obstacle 최소 거리')
    args = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    home = Path(os.environ.get('STABLEWM_HOME', Path.home() / '.stable-wm'))
    det, fc, meta = od.load(home / 'checkpoints' / args.detector, dev)
    env = gym.make('iawm/CollisionTwoRoom-v1', num_obstacles=args.num_obstacles, obstacle_radius=22.0,
                   obstacle_min_sep=args.min_sep).unwrapped
    pair0, pair10, retries = [], [], []
    rec = []  # (missed, d_obstacle, d_agent, d_border)
    vel_err = []
    first_vel = []
    rng = np.random.default_rng(0)
    for ep in range(args.episodes):
        env.reset(seed=ep)
        # 첫 계획과 같은 조건: 1 step 전 화면은 render_past(1)로 만든다(eval.py HistoryPolicy)
        cur0 = np.asarray(env.render())
        past = np.asarray(env.render_past(1))
        P, V, M = det.detect(torch.from_numpy(past[None]).to(dev), torch.from_numpy(cur0[None]).to(dev), thr=args.thr)
        tp0, tv0 = oo.current_state(env)
        retries.append(getattr(env, 'spawn_retries', 0))
        q = tp0.reshape(-1, 2)
        if len(q) > 1:
            pair0.append(min(np.linalg.norm(q[i] - q[j]) for i in range(len(q)) for j in range(i + 1, len(q))))
        pe, ve, _, _ = od.match_error(P[0].cpu().numpy(), V[0].cpu().numpy(), M[0].cpu().numpy(), tp0.reshape(-1, 2), tv0.reshape(-1, 2))
        first_vel += ve
        prev = cur0
        for t in range(args.steps):
            env.step(rng.uniform(-1, 1, 2).astype(np.float32))
            cur = np.asarray(env.render())
            P, V, M = det.detect(torch.from_numpy(prev[None]).to(dev), torch.from_numpy(cur[None]).to(dev), thr=args.thr)
            P, V, M = P[0].cpu().numpy(), V[0].cpu().numpy(), M[0].cpu().numpy()
            tp, tv = oo.current_state(env)
            tp, tv = tp.reshape(-1, 2), tv.reshape(-1, 2)
            det_idx = np.nonzero(M)[0]
            pairs = dict(od._assign(tp, P[det_idx], 15.0))
            agent = np.asarray(env.agent_position.detach().cpu().numpy() if torch.is_tensor(env.agent_position) else env.agent_position).reshape(-1)[:2]
            for k in range(len(tp)):
                d_ob = min([np.linalg.norm(tp[k] - tp[j]) for j in range(len(tp)) if j != k] or [999])
                d_ag = float(np.linalg.norm(tp[k] - agent))
                d_bd = float(min(tp[k].min(), (od.IMG - tp[k]).min()))
                rec.append((k not in pairs, d_ob, d_ag, d_bd))
                if k in pairs:
                    vel_err.append(float(np.linalg.norm(tv[k] - V[det_idx[pairs[k]]])))
            if t == 9 and len(tp) > 1:
                pair10.append(min(np.linalg.norm(tp[i] - tp[j]) for i in range(len(tp)) for j in range(i + 1, len(tp))))
            prev = cur
    r = np.array(rec)
    if pair0:
        p0, p10 = np.array(pair0), np.array(pair10)
        print(f'DIAG spawn: min pair dist t0 mean {p0.mean():.1f} frac<44 {(p0 < 44).mean():.3f} | t10 mean {p10.mean():.1f} '
              f'frac<44 {(p10 < 44).mean():.3f} | spawn retries mean {np.mean(retries):.2f}')
    miss = r[:, 0] > 0
    print(f'DIAG n {len(r)} miss {miss.mean():.4f} vel_mae {np.mean(vel_err):.3f} | first step (render_past prev) vel_mae {np.mean(first_vel):.3f} n {len(first_vel)}')
    for name, col, edges in (('obstacle_dist', 1, [0, 30, 44, 60, 999]), ('agent_dist', 2, [0, 20, 30, 45, 999]),
                             ('border_dist', 3, [0, 30, 40, 60, 999])):
        parts = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (r[:, col] >= lo) & (r[:, col] < hi)
            if sel.any():
                parts.append(f'[{lo},{hi}) n {sel.sum()} miss {miss[sel].mean():.3f}')
        print(f'DIAG {name}: ' + ' | '.join(parts))


if __name__ == '__main__':
    main()
