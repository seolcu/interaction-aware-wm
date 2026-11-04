"""학습된 gate 모델을 고정하고 h_int만 새로 붙여 z^c 잔차를 맞힐 수 있는지 본다 (E20 진단).

E16에서 정답 state를 h_int에 넣어도 contact 구간 이득이 1.4%에 그쳤다. 원인 후보를 나눈다.
- 함께 학습(encoder, g, target이 같이 움직임)과 작은 lr 때문인가
- h_int 구조(AdaLN-zero transformer + zero-init 출력) 때문인가
- 잔차 자체를 정답 state로도 맞히기 어려운가

고정한 모델에서 clip마다 z^c, z^u, action embedding, g의 예측(base), 다음 frame z^c(target),
정답 state, contact 라벨을 뽑고, 아래 head들을 잔차 (target - base)에 따로 학습해 test episode에서
상대 이득 (err_off - err_on) / err_off를 contact별로 잰다. 출력 형태는 학습과 같게 int_scale * tanh다.

- own: 모델에 들어 있는 학습된 h_int (학습하지 않고 평가만)
- tf_oracle: 같은 구조의 새 transformer h_int, 입력 [z^c, z^u, oracle]
- tf_latent: 같은 구조, 입력 [z^c, z^u] (실제 gate 모델과 같은 입력)
- mlp_oracle: frame별 MLP, 입력 [z^c, z^u, oracle, action]
- mlp_latent: frame별 MLP, 입력 [z^c, z^u, action] (정답 state 없음)
- mlp_oracle_only: frame별 MLP, 입력 [oracle, action]

사용:
    python scripts/fit_hint_frozen.py --ckpt e16_oracle/weights_epoch_4.pt --data collision_tworoom_contact.h5
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT), str(ROOT / 'scripts')]

import numpy as np
import torch
from torch import nn

from analyze_gate import action_stats, iter_clips, to_model_pixels, IMAGENET_MEAN, IMAGENET_STD


@torch.no_grad()
def extract(model, f, episodes, device, oracle, batch=256, distractors=2, frameskip=5, history=3):
    act_mean, act_std = action_stats(f)
    pr = model.predictor
    d_c = model.dim_c
    keys = ('zc', 'zu', 'act', 'base', 'tgt', 'orc', 'own', 'lab')
    out = {k: [] for k in keys}
    bufs = {'px': [], 'act': [], 'lab': [], 'orc': []}

    def flush():
        if not bufs['px']:
            return
        px = np.concatenate(bufs['px']); act = torch.from_numpy(np.concatenate(bufs['act'])).to(device)
        lab = np.concatenate(bufs['lab']); orc = torch.from_numpy(np.concatenate(bufs['orc'])).to(device)
        for v in bufs.values():
            v.clear()
        for i in range(0, len(px), batch):
            pixels = to_model_pixels(px[i:i + batch], device)
            if distractors > 0:
                from iawm.distractors import DistractorConfig, overlay, sample_params
                dcfg = DistractorConfig(num=distractors, img_size=pixels.size(-1))
                params = sample_params(pixels.size(0), dcfg, device=device)
                t = torch.arange(pixels.size(1), device=device).float() * frameskip
                pixels = overlay(pixels, params, t, dcfg, IMAGENET_MEAN.view(1, 3), IMAGENET_STD.view(1, 3))
            o = model.encode({'pixels': pixels, 'action': act[i:i + batch]})
            emb, act_emb = o['emb'], o['act_emb']
            if getattr(pr, 'int_oracle_dim', 0) > 0:
                pr.oracle = orc[i:i + batch]
            model.predict(emb[:, :history], act_emb[:, :history])
            z = emb[:, :history].float()
            out['zc'].append(z[..., :d_c].cpu()); out['zu'].append(z[..., d_c:].cpu())
            out['act'].append(act_emb[:, :history].float().cpu())
            out['base'].append(pr.last_pred_c_base.float().cpu())
            out['own'].append(pr.last_delta.float().cpu())
            out['tgt'].append(emb[:, 1:history + 1, :d_c].float().cpu())
            out['orc'].append(orc[i:i + batch, :history].float().cpu())
            out['lab'].append(torch.from_numpy(lab[i:i + batch]))

    n = 0
    for px, act, lab, ep, orc in iter_clips(f, episodes, history + 1, frameskip, frameskip, act_mean, act_std, oracle):
        bufs['px'].append(px); bufs['act'].append(act); bufs['lab'].append(lab); bufs['orc'].append(orc)
        n += len(px)
        if n >= 4 * batch:
            flush(); n = 0
    flush()
    return {k: torch.cat(v) for k, v in out.items()}


class FrameMLP(nn.Module):
    def __init__(self, in_dim, d_c, hidden=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, d_c))

    def forward(self, d):
        return self.net(torch.cat(d['inputs'], -1))


class TFHead(nn.Module):
    def __init__(self, in_dim, d_c, num_frames=3):
        super().__init__()
        from stable_worldmodel.wm.lewm.module import Predictor
        self.inp = nn.Linear(in_dim, d_c)
        self.tf = Predictor(num_frames=num_frames, input_dim=d_c, hidden_dim=128, output_dim=d_c,
                            depth=2, heads=4, dim_head=32, mlp_dim=512, dropout=0.1)
        self.out = nn.Linear(d_c, d_c)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, d):
        return self.out(self.tf(self.inp(torch.cat(d['inputs'], -1)), d['act']))


def rel_gain(base, delta, tgt, lab):
    off = (base - tgt).pow(2).mean(-1)
    on = (base + delta - tgt).pow(2).mean(-1)
    g = ((off - on) / off.clamp(min=1e-8))
    lab = lab.bool()
    return g[~lab].mean().item(), g[lab].mean().item(), (off[lab].mean() - on[lab].mean()).item() / off[lab].mean().item()


def fit(head, tr, te, inputs_fn, device, steps, lr, scale, bs=512):
    head = head.to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-3)
    N = len(tr['zc'])
    to = lambda d, idx: {k: v[idx].to(device) for k, v in d.items()}
    for s in range(steps):
        b = to(tr, torch.randint(0, N, (bs,)))
        delta = scale * torch.tanh(head({'inputs': inputs_fn(b), 'act': b['act']}))
        loss = (b['base'] + delta - b['tgt']).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    head.eval()
    deltas = []
    with torch.no_grad():
        for i in range(0, len(te['zc']), 2048):
            b = to(te, slice(i, i + 2048))
            deltas.append(scale * torch.tanh(head({'inputs': inputs_fn(b), 'act': b['act']})).cpu())
    return torch.cat(deltas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data', default='collision_tworoom_contact.h5')
    ap.add_argument('--train-episodes', type=int, default=3000)
    ap.add_argument('--test-episodes', type=int, default=500)
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--lrs', default='1e-3,5e-5')
    ap.add_argument('--distractors', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--heads', default=None, help='쉼표로 고른 head만 (기본: 전부)')
    args = ap.parse_args()

    import h5py
    import hdf5plugin  # noqa: F401
    import stable_worldmodel as swm

    torch.manual_seed(args.seed)
    device = 'cuda'
    model = swm.wm.utils.load_pretrained(args.ckpt).to(device).eval()
    scale = getattr(model.predictor, 'int_scale', 1.0)
    path = Path(swm.data.utils.get_cache_dir()) / 'datasets' / args.data
    with h5py.File(path, 'r') as f:
        o_all = f['oracle_state'][:].astype(np.float64)
        ok = o_all[~np.isnan(o_all).any(axis=1)]
        oracle = np.nan_to_num(((o_all - ok.mean(0)) / ok.std(0, ddof=1)).astype(np.float32), nan=0.0)
        n_ep = len(f['ep_len'])
        tr = extract(model, f, range(0, args.train_episodes), device, oracle, distractors=args.distractors)
        te = extract(model, f, range(n_ep - args.test_episodes, n_ep), device, oracle, distractors=args.distractors)
    d_c = model.dim_c
    d_u = te['zu'].size(-1)
    print(f'ckpt={args.ckpt} train clips={len(tr["zc"])} test clips={len(te["zc"])} '
          f'contact ratio test={te["lab"].float().mean():.3f} int_scale={scale}')
    print(f'{"head":<18}{"lr":>8}{"non-contact":>13}{"contact":>10}{"contact(pooled)":>17}')
    nc, c, cp = rel_gain(te['base'], te['own'], te['tgt'], te['lab'])
    print(f'{"own (trained)":<18}{"-":>8}{nc*100:>12.2f}%{c*100:>9.2f}%{cp*100:>16.2f}%')
    heads = {
        'tf_oracle': (lambda: TFHead(d_c + d_u + 8, d_c), lambda b: [b['zc'], b['zu'], b['orc']]),
        'tf_latent': (lambda: TFHead(d_c + d_u, d_c), lambda b: [b['zc'], b['zu']]),
        'mlp_oracle': (lambda: FrameMLP(d_c + d_u + 8 + d_c, d_c), lambda b: [b['zc'], b['zu'], b['orc'], b['act']]),
        'mlp_latent': (lambda: FrameMLP(d_c + d_u + d_c, d_c), lambda b: [b['zc'], b['zu'], b['act']]),
        'mlp_oracle_only': (lambda: FrameMLP(8 + d_c, d_c), lambda b: [b['orc'], b['act']]),
    }
    if args.heads:
        heads = {k: v for k, v in heads.items() if k in args.heads.split(',')}
    for lr in [float(x) for x in args.lrs.split(',')]:
        for name, (make, inp) in heads.items():
            torch.manual_seed(args.seed)
            delta = fit(make(), tr, te, inp, device, args.steps, lr, scale)
            nc, c, cp = rel_gain(te['base'], delta, te['tgt'], te['lab'])
            print(f'{name:<18}{lr:>8.0e}{nc*100:>12.2f}%{c*100:>9.2f}%{cp*100:>16.2f}%', flush=True)


if __name__ == '__main__':
    main()
