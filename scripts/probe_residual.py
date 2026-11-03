"""CF-JEPA의 z^c 예측 잔차를 frozen latent에서 맞힐 수 있는지 확인한다 (E9, 학습된 latent 위의 oracle 실험).

E9에서 h_int는 α=1로 직접 학습해도 contact 구간 z^c 오차를 0.4%밖에 줄이지 못했다. 이것이
"latent에 정보가 부족해서"인지 "h_int가 배우지 못해서"인지 가른다.

학습된 checkpoint를 고정하고, clip마다 마지막 position(history 3 frame을 모두 본 예측)에서
    잔차 r = z^c_{t+1} - (h_int를 더하기 전 z^c 예측)
를 구한다. 그리고 큰 MLP(early stopping)로 r을 맞힌다. 입력은 세 가지다.
- zc:      z^c_{t-2..t} + action_t                         (agent 쪽 latent만)
- zc+zu:   z^c_{t-2..t} + z^u_{t-2..t} + action_t           (h_int가 볼 수 있는 정보)
- oracle:  agent 위치_{t-2..t}, obstacle 위치·속도_t + action_t  (ground-truth state, E2와 같은 역할)
Episode 기준 80/20 분할의 test에서 contact / non-contact별 오차 감소율 1 - |r - r̂|² / |r|²을 보고한다.

사용:
    python scripts/probe_residual.py --ckpt cfjepa_collision_n2/weights_epoch_10.pt
    python scripts/probe_residual.py --ckpt cfjepa_collision_n2/weights_epoch_10.pt --episodes 2000
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT), str(ROOT / 'scripts')]

import numpy as np
import torch

from analyze_gate import action_stats, episode_offsets, to_model_pixels
from probe_latent import agent_pos, mlp_fit_predict

NUM_STEPS, FS, HIST = 4, 5, 3


def clip_rows(f, episodes, act_mean, act_std):
    """episode마다 clip(시작 row s, 5 step 간격)을 만든다. 마지막 transition(frame s+10 -> s+15)이 대상이다."""
    off, ep_len = episode_offsets(f)
    span = NUM_STEPS * FS
    for ep in episodes:
        o, L = int(off[ep]), int(ep_len[ep])
        if L < span:
            continue
        px = f['pixels'][o:o + L]
        act = np.nan_to_num((f['action'][o:o + L].astype(np.float32) - act_mean) / act_std, nan=0.0)
        con = f['contact'][o:o + L].astype(bool)
        ag = agent_pos(f, o, L)
        # obstacle 전부 (E16 수정: 예전에는 첫 obstacle만 썼다. obstacle이 여러 개인 data(E15)에서 oracle이 약했던 원인)
        obs = f['obstacle_pos'][o:o + L].reshape(L, -1).astype(np.float32)
        vel = f['obstacle_vel'][o:o + L].reshape(L, -1).astype(np.float32)
        starts = np.arange(0, L - span + 1, FS)
        fr = starts[:, None] + FS * np.arange(NUM_STEPS)[None]           # frame rows
        t = fr[:, HIST - 1]                                              # 마지막 입력 frame row
        lab = np.array([con[s + 1:s + FS + 1].any() for s in t])         # transition t -> t+5의 contact
        oracle = np.concatenate([
            ag[fr[:, :HIST]].reshape(len(starts), -1),                   # agent 위치 t-2..t
            obs[t], vel[t],                                              # obstacle 위치·속도 t
            act[t[:, None] + np.arange(FS)[None]].reshape(len(starts), -1),  # action t..t+4
        ], 1)
        yield (px[fr], act[starts[:, None] + np.arange(span)[None]].reshape(len(starts), NUM_STEPS, -1),
               lab, oracle, np.full(len(starts), ep))


@torch.no_grad()
def collect(model, f, episodes, device, batch=128):
    act_mean, act_std = action_stats(f)
    d_c = model.dim_c
    Z, R, A, LAB, ORC, EP = [], [], [], [], [], []
    for px, act, lab, orc, ep in clip_rows(f, episodes, act_mean, act_std):
        for i in range(0, len(px), batch):
            a = torch.from_numpy(act[i:i + batch]).to(device)
            out = model.encode({'pixels': to_model_pixels(px[i:i + batch], device), 'action': a})
            emb, act_emb = out['emb'], out['act_emb']
            pred = model.predict(emb[:, :HIST], act_emb[:, :HIST])
            base = getattr(model.predictor, 'last_pred_c_base', None)
            base = (base if base is not None else pred[..., :d_c])[:, -1].float()
            R.append((emb[:, HIST, :d_c].float() - base).cpu().numpy())
            Z.append(emb[:, :HIST].float().cpu().numpy())                 # (b, 3, 192)
            A.append(a[:, HIST - 1].cpu().numpy())                          # action t..t+4 (10)
        LAB.append(lab), ORC.append(orc), EP.append(ep)
    return tuple(map(np.concatenate, (Z, R, A, LAB, ORC, EP)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data', default='collision_tworoom.h5')
    ap.add_argument('--episodes', type=int, default=1500, help='data의 마지막 N episode')
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--steps', type=int, default=4000, help='MLP 학습 step')
    args = ap.parse_args()

    import h5py
    import hdf5plugin  # noqa: F401
    import stable_worldmodel as swm

    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = swm.wm.utils.load_pretrained(args.ckpt).to(device).eval()
    d_c = model.dim_c
    path = Path(swm.data.utils.get_cache_dir()) / 'datasets' / args.data
    with h5py.File(path, 'r') as f:
        n_ep = len(f['ep_len'])
        episodes = range(max(0, n_ep - args.episodes), n_ep)
        Z, R, A, lab, orc, ep = collect(model, f, episodes, device, args.batch)

    eps = np.unique(ep)
    te = np.isin(ep, eps[int(0.8 * len(eps)):])
    tr = ~te
    n = len(Z)
    inputs = {
        'zc': np.concatenate([Z[:, :, :d_c].reshape(n, -1), A], 1),
        'zc+zu': np.concatenate([Z.reshape(n, -1), A], 1),
        'oracle': orc,
    }
    rm, rs = R[tr].mean(0), R[tr].std(0) + 1e-8
    Rn = ((R - rm) / rs).astype(np.float32)
    print(f'ckpt={args.ckpt} episodes={len(episodes)} transitions: train {tr.sum()} test {te.sum()} '
          f'contact ratio {lab.mean():.4f}')
    e0 = (R[te] ** 2).mean(1)
    c = lab[te]
    print(f'잔차 크기 |r|² (= CF-JEPA z^c 오차): non-contact {e0[~c].mean():.5f}  contact {e0[c].mean():.5f}  '
          f'ratio {e0[c].mean() / e0[~c].mean():.2f}')
    # 잔차의 평균(bias)만 맞혀도 |r|²은 줄어든다. 그건 interaction과 무관하므로, group별 평균 잔차
    # (train에서 구함)를 예측하는 기준선 대비 감소율을 주 지표로 본다.
    e_const = np.zeros_like(e0)
    for m_tr, m_te in ((lab[tr], c), (~lab[tr], ~c)):
        e_const[m_te] = ((R[te][m_te] - R[tr][m_tr].mean(0)) ** 2).mean(1)
    print(f'group별 평균 잔차를 빼고 남는 오차: non-contact {e_const[~c].mean():.5f}  contact {e_const[c].mean():.5f}')
    print('오차 감소율 (높을수록 그 입력으로 잔차를 맞힐 수 있다)')
    print('  A = 1 - |r - r̂|² / |r|²                  (CF-JEPA 오차 대비)')
    print('  B = 1 - |r - r̂|² / |r - group 평균 r|²   (평균 보정 이후 남는 오차 대비, 주 지표)')
    print(f'{"input":10s}{"non-contact A / B":>22s}{"contact A / B":>20s}')
    for name, X in inputs.items():
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xn = ((X - mu) / sd).astype(np.float32)
        pred = mlp_fit_predict(Xn[tr], Rn[tr], Xn[te], device, steps=args.steps, hidden=512) * rs + rm
        e1 = ((R[te] - pred) ** 2).mean(1)
        red = lambda m, base: 1 - e1[m].sum() / base[m].sum()  # noqa: E731
        print(f'{name:10s}{red(~c, e0):>12.3f} / {red(~c, e_const):<7.3f}{red(c, e0):>10.3f} / {red(c, e_const):<7.3f}')


if __name__ == '__main__':
    main()
