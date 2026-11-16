"""학습된 CF-JEPA를 고정하고 interaction module(h_int, int_in, int_out, gate)만 학습한다 (E39).

`scripts/train.py +init_from=... +freeze_base=true`(E35)와 같은 일을 하지만, encoder가 고정이므로 모든 frame의
latent를 한 번만 계산해 두고(cache) 그 위에서 학습한다. E35는 epoch마다 화면을 다시 encoding해서 L4에서 epoch당
약 55분이 걸렸다. 여기서는 encoding 한 번(약 10분) 뒤 epoch당 1분 안팎이다.

E35와 같은 것:
- 모델: base(CF-JEPA) weights를 gate 모델에 넣고 interaction module만 학습한다. gate를 끄면 base와 정확히 같다.
- loss: fwd(α·h_int를 더한 예측의 오차) + int_aux_weight · (α = 1 예측의 z^c 오차) + gate BCE(contact 라벨).
  scripts/train.py의 [interaction gate] 블록(gate_objective=supervised)과 같은 식이다. base에만 걸리는
  loss(SIGReg, inverse, adversarial)는 고정된 parameter에만 영향을 주므로 뺀다.
- 고정한 부분은 eval mode(dropout 끔, BatchNorm 통계 고정), 학습하는 부분만 train mode.
E35와 다른 것:
- data: base가 학습에 쓰지 않은 data로 학습해야 한다. E35는 base의 학습 data(같은 seed로 다시 수집)를 써서
  h_int가 "base가 외운 data에서 남긴 오차"를 배웠다. 수집 seed를 바꾼 data를 --data로 준다.
- latent는 float32로 계산한다(train.py는 bf16-mixed). planning 평가가 float32라 그쪽에 맞춘다.
- --oracle obs: h_int 입력에 정답 obstacle 상태(iawm.obs_oracle)를 더한다(상한 실험).
- 검증: data의 마지막 --val-episodes episode는 학습에 쓰지 않고, epoch마다 contact / non-contact z^c 오차를 잰다.
  검증 오차(학습된 α로 더한 예측)가 가장 낮은 epoch를 weights_best.pt로 남긴다.
- --test-data: 따로 수집한 data에서 최종 수치를 잰다(RESULT_TEST 줄. pipeline이 읽는다).

출력: $STABLEWM_HOME/checkpoints/<out>/weights_epoch_<k>.pt, weights_best.pt, config.json, metrics.json.
swm.wm.utils.load_pretrained('<out>/weights_best.pt')로 불러 scripts/eval.py, analyze_gate.py에 그대로 쓴다.

사용:
    python scripts/train_int_frozen.py --base e32_cf/weights_epoch_10.pt --data r1y.h5 --out e39_lat
    python scripts/train_int_frozen.py --base e32_cf/weights_epoch_10.pt --data r1y.h5 --out e39_obs --oracle obs
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT), str(ROOT / 'scripts')]

import numpy as np
import torch

from add_contact_labels import contact_next_labels
from analyze_gate import action_stats, episode_offsets, to_model_pixels

INT_RE = re.compile(r'^predictor\.(h_int|int_in|int_out|gate)(\.|$)')
FS, NUM_STEPS, HIST = 5, 4, 3  # frameskip, clip 길이(history + 1), history. 학습 config와 같다


def gate_config(base_cfg: dict, args) -> dict:
    """base(CF-JEPA) config를 같은 크기의 gate 모델 config로 바꾼다. E35의 e35_fres config와 같은 형태다."""
    cfg = json.loads(json.dumps(base_cfg))
    assert cfg['_target_'] == 'iawm.cfjepa.CFJEPA', f"base는 CF-JEPA여야 한다: {cfg['_target_']}"
    cfg['_target_'] = 'iawm.cfjepa_gate.CFJEPA'
    p = cfg['predictor']
    p['_target_'] = 'iawm.cfjepa_gate.InteractionGatedPredictor'
    p.update(gate_mode='learned', zu_stopgrad=False, gate_objective='supervised', int_input=args.int_input,
             int_hidden_dim=args.int_hidden_dim, int_depth=args.int_depth, int_heads=args.int_heads,
             int_dim_head=32, int_mlp_dim=args.int_mlp_dim, int_scale=1.0, gate_hidden_dim=256, gate_init_bias=2.0)
    if args.oracle == 'obs':
        p.update(int_oracle_dim=6 * args.num_obstacles, int_oracle_key='obs_oracle')
    cfg.update(int_aux_weight=args.int_aux_weight, gate_bce_weight=1.0, int_lr=args.lr)
    return cfg


@torch.no_grad()
def frame_latents(model, f, device, cache: Path | None, batch=500, chunk=2000):
    """file의 모든 row(frame)의 latent (N, D) float32. cache가 있으면 읽는다."""
    n = len(f['pixels'])
    if cache is not None and cache.exists():
        z = np.load(cache)
        if len(z) == n:
            print(f'latents: cache {cache} ({z.shape})', flush=True)
            return z
    out, t0 = [], time.time()
    for i in range(0, n, chunk):
        px = f['pixels'][i:i + chunk]
        for j in range(0, len(px), batch):
            pixels = to_model_pixels(px[j:j + batch][:, None], device)  # (b, 1, 3, H, W)
            out.append(model.encode({'pixels': pixels})['emb'][:, 0].float().cpu().numpy())
        if (i // chunk) % 25 == 0:
            el = time.time() - t0
            print(f'latents: {i + len(px)}/{n} rows, {el:.0f}s, {(i + len(px)) / max(el, 1e-6):.0f} rows/s', flush=True)
    z = np.concatenate(out)
    if cache is not None:  # 임시 파일에 쓰고 이름을 바꾼다: 같이 도는 다른 학습이 쓰다 만 파일을 읽지 않게 한다
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_name(f'{cache.stem}.tmp{int(time.time() * 1e6)}.npy')
        np.save(tmp, z)
        tmp.replace(cache)
    return z


def load_rows(model, path: Path, device, cache: Path | None, oracle: str):
    """학습에 쓰는 row별 tensor를 GPU에 올린다. 정의는 analyze_gate.iter_clips, add_contact_labels와 같다."""
    import h5py
    import hdf5plugin  # noqa: F401

    from iawm import obs_oracle

    with h5py.File(path, 'r') as f:
        off, ep_len = episode_offsets(f)
        z = frame_latents(model, f, device, cache)
        mean, std = action_stats(f)
        act = np.nan_to_num((f['action'][:].astype(np.float32) - mean) / std, nan=0.0).astype(np.float32)
        contact = f['contact'][:].reshape(len(f['contact']), -1)[:, 0]
        cn = contact_next_labels(contact, off, ep_len, FS)  # row j: frame j -> j+5 사이 contact
        orc = obs_oracle.dataset_column(f, FS) if oracle == 'obs' else None
    rows = {'z': torch.from_numpy(z).to(device), 'act': torch.from_numpy(act).to(device),
            'cn': torch.from_numpy(cn).to(device)}
    if orc is not None:
        rows['orc'] = torch.from_numpy(orc).to(device)
    return rows, off, ep_len


def clip_starts(off, ep_len, episodes, stride):
    """clip 시작 row. clip은 frame row s, s+5, s+10, s+15와 action row s..s+19다(학습 loader와 같다)."""
    span = NUM_STEPS * FS
    out = [off[ep] + np.arange(0, ep_len[ep] - span + 1, stride) for ep in episodes if ep_len[ep] >= span]
    return np.concatenate(out).astype(np.int64)


def forward(model, rows, s):
    """clip 시작 row s (B,)에 대한 예측. 반환: pred, tgt, base, delta, logit, alpha, label (모두 position 0..2)."""
    pr = model.predictor
    fr = s[:, None] + FS * torch.arange(NUM_STEPS, device=s.device)[None]            # (B, 4)
    emb = rows['z'][fr]                                                               # (B, 4, D)
    act = rows['act'][s[:, None] + torch.arange(NUM_STEPS * FS, device=s.device)[None]].reshape(len(s), NUM_STEPS, -1)
    act_emb = model.action_encoder(act)
    if 'orc' in rows:
        pr.oracle = rows['orc'][fr[:, :HIST]]
    pred = model.predict(emb[:, :HIST], act_emb[:, :HIST])
    return {'pred': pred, 'tgt': emb[:, 1:HIST + 1], 'base': pr.last_pred_c_base, 'delta': pr.last_delta,
            'logit': pr.last_logit.squeeze(-1), 'alpha': pr.last_alpha.squeeze(-1), 'lab': rows['cn'][fr[:, :HIST]] > 0.5}


@torch.no_grad()
def evaluate(model, rows, starts, d_c, batch=4096):
    """contact / non-contact별 z^c 오차. off = base(CF-JEPA), on = α 1, learned = 학습된 α."""
    model.eval()
    acc = {k: [] for k in ('off', 'on', 'learned', 'alpha', 'lab')}
    for i in range(0, len(starts), batch):
        o = forward(model, rows, starts[i:i + batch])
        tgt_c = o['tgt'][..., :d_c].float()
        acc['off'].append((o['base'].float() - tgt_c).pow(2).mean(-1).flatten())
        acc['on'].append((o['base'].float() + o['delta'].float() - tgt_c).pow(2).mean(-1).flatten())
        acc['learned'].append((o['pred'][..., :d_c].float() - tgt_c).pow(2).mean(-1).flatten())
        acc['alpha'].append(o['alpha'].float().flatten())
        acc['lab'].append(o['lab'].flatten())
    a = {k: torch.cat(v) for k, v in acc.items()}
    c = a['lab']
    m = {'n': int(len(c)), 'contact_ratio': c.float().mean().item()}
    for k in ('off', 'on', 'learned'):
        m[f'c_{k}'], m[f'nc_{k}'], m[f'all_{k}'] = a[k][c].mean().item(), a[k][~c].mean().item(), a[k].mean().item()
    m['c_red_on'] = 1 - m['c_on'] / m['c_off']            # contact 오차 감소율, α = 1
    m['c_red_learned'] = 1 - m['c_learned'] / m['c_off']  # contact 오차 감소율, 학습된 α (planning이 쓰는 예측)
    m['nc_red_learned'] = 1 - m['nc_learned'] / m['nc_off']
    m['alpha_c'], m['alpha_nc'] = a['alpha'][c].mean().item(), a['alpha'][~c].mean().item()
    try:
        from sklearn.metrics import roc_auc_score
        m['auroc'] = float(roc_auc_score(c.cpu().numpy(), a['alpha'].cpu().numpy()))
    except Exception:
        m['auroc'] = float('nan')
    return m


def fmt(m):
    return (f"contact off/on/learned {m['c_off']:.5f}/{m['c_on']:.5f}/{m['c_learned']:.5f} "
            f"(red on {m['c_red_on']:+.3f} learned {m['c_red_learned']:+.3f}) | non-contact off/learned "
            f"{m['nc_off']:.5f}/{m['nc_learned']:.5f} ({m['nc_red_learned']:+.3f}) | alpha c/nc {m['alpha_c']:.2f}/{m['alpha_nc']:.2f} "
            f"auroc {m['auroc']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True, help='고정할 CF-JEPA checkpoint ($STABLEWM_HOME/checkpoints 기준)')
    ap.add_argument('--data', required=True, help='학습 data (base가 학습에 쓰지 않은 것)')
    ap.add_argument('--test-data', default=None, help='최종 수치를 잴 따로 수집한 data')
    ap.add_argument('--out', required=True, help='출력 checkpoint 폴더 이름')
    ap.add_argument('--oracle', default='none', choices=['none', 'obs'])
    ap.add_argument('--num-obstacles', type=int, default=1)
    ap.add_argument('--int-input', default='resolve', choices=['state', 'resolve'])
    ap.add_argument('--int-depth', type=int, default=2)
    ap.add_argument('--int-hidden-dim', type=int, default=128)
    ap.add_argument('--int-heads', type=int, default=4)
    ap.add_argument('--int-mlp-dim', type=int, default=512)
    ap.add_argument('--int-aux-weight', type=float, default=1.0)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=1e-3)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--stride', type=int, default=1, help='학습 clip 시작 간격 (학습 loader는 1)')
    ap.add_argument('--val-episodes', type=int, default=500)
    ap.add_argument('--max-episodes', type=int, default=None, help='smoke test용: 앞에서 이만큼만 쓴다')
    ap.add_argument('--save-every', type=int, default=10)
    ap.add_argument('--cache-dir', default=None, help='frame latent cache 폴더 (기본: $STABLEWM_HOME/latent_cache)')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    import hydra
    import stable_worldmodel as swm

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    home = Path(swm.data.utils.get_cache_dir())
    ck = home / 'checkpoints'
    base_path = ck / args.base
    cfg = gate_config(json.load(open(base_path.parent / 'config.json')), args)
    model = hydra.utils.instantiate(cfg)
    sd = torch.load(base_path, map_location='cpu', weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    bad = [k for k in missing if not INT_RE.match(k)]
    assert not unexpected and not bad, f'base mismatch: unexpected={unexpected[:5]} missing={bad[:5]}'
    n_train = 0
    for name, p in model.named_parameters():
        p.requires_grad_(bool(INT_RE.match(name)))
        n_train += p.numel() if p.requires_grad else 0
    model = model.to(device).eval()
    pr, d_c = model.predictor, model.dim_c
    print(f'base={args.base} loaded {len(sd)} tensors, new={len(missing)}, trainable={n_train / 1e6:.3f}M '
          f'oracle={args.oracle} int_input={args.int_input} depth={args.int_depth} hidden={args.int_hidden_dim}', flush=True)

    cache_dir = Path(args.cache_dir) if args.cache_dir else home / 'latent_cache'
    tag = args.base.replace('/', '_').replace('.pt', '')

    def rows_of(name):
        return load_rows(model, home / 'datasets' / name, device, cache_dir / f'{tag}__{Path(name).stem}.npy', args.oracle)

    rows, off, ep_len = rows_of(args.data)
    n_ep = len(ep_len) if args.max_episodes is None else min(args.max_episodes, len(ep_len))
    n_val = min(args.val_episodes, n_ep // 5)
    tr_starts = torch.from_numpy(clip_starts(off, ep_len, range(0, n_ep - n_val), args.stride)).to(device)
    va_starts = torch.from_numpy(clip_starts(off, ep_len, range(n_ep - n_val, n_ep), FS)).to(device)
    print(f'data={args.data} episodes train {n_ep - n_val} val {n_val} clips train {len(tr_starts)} val {len(va_starts)}', flush=True)

    # 확인: 학습 전에는 int_out이 0이라 예측이 base와 같아야 한다 (gate off = CF-JEPA)
    m0 = evaluate(model, rows, va_starts, d_c)
    assert abs(m0['c_learned'] - m0['c_off']) < 1e-7 and abs(m0['nc_learned'] - m0['nc_off']) < 1e-7, m0
    print(f'epoch 0 (= CF-JEPA) val: {fmt(m0)}', flush=True)

    train_mods = [pr.h_int, pr.int_in, pr.int_out, pr.gate]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(tr_starts) // args.batch
    total, warm = args.epochs * steps_per_epoch, min(500, steps_per_epoch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: (t + 1) / warm if t < warm else 0.5 * (1 + np.cos(np.pi * (t - warm) / max(1, total - warm))))
    out_dir = ck / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(cfg, open(out_dir / 'config.json', 'w'), indent=2)
    history, best, best_epoch, t0 = [dict(m0, epoch=0)], m0['all_learned'], 0, time.time()
    bce_fn = torch.nn.functional.binary_cross_entropy_with_logits
    for epoch in range(1, args.epochs + 1):
        model.eval()
        for mod in train_mods:
            mod.train()
        perm = tr_starts[torch.randperm(len(tr_starts), device=device)]
        tot = {'fwd': 0.0, 'aux': 0.0, 'bce': 0.0}
        for b in range(steps_per_epoch):
            o = forward(model, rows, perm[b * args.batch:(b + 1) * args.batch])
            tgt_c = o['tgt'][..., :d_c].float().detach()
            fwd = (o['pred'] - o['tgt']).pow(2).mean()
            aux = (o['base'].float().detach() + o['delta'].float() - tgt_c).pow(2).mean()
            bce = bce_fn(o['logit'].float(), o['lab'].float())
            loss = fwd + args.int_aux_weight * aux + bce
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            for k, v in (('fwd', fwd), ('aux', aux), ('bce', bce)):
                tot[k] += v.item()
        m = evaluate(model, rows, va_starts, d_c)
        m.update(epoch=epoch, **{f'train_{k}': v / steps_per_epoch for k, v in tot.items()})
        history.append(m)
        flag = ''
        if m['all_learned'] < best:
            best, best_epoch, flag = m['all_learned'], epoch, ' *best'
            torch.save(model.state_dict(), out_dir / 'weights_best.pt')
            # 학습한 부분만 따로 남긴다(수 MB). 세션을 급히 닫을 때 이것만 받아도 base와 합쳐 복원할 수 있다.
            torch.save({k: v for k, v in model.state_dict().items() if INT_RE.match(k)}, out_dir / 'int_only_best.pt')
        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save(model.state_dict(), out_dir / f'weights_epoch_{epoch}.pt')
        print(f"epoch {epoch} ({time.time() - t0:.0f}s) train fwd/aux/bce {m['train_fwd']:.5f}/{m['train_aux']:.5f}/{m['train_bce']:.4f} "
              f"val: {fmt(m)}{flag}", flush=True)
    if best_epoch == 0:  # 한 번도 base보다 좋아지지 않았다: 마지막 epoch를 best로 남겨 뒤 단계가 돌 수 있게 한다
        torch.save(model.state_dict(), out_dir / 'weights_best.pt')
    mb = history[best_epoch]
    print(f"RESULT_VAL {args.out} best_epoch {best_epoch} contact_red_learned {mb['c_red_learned']:.4f} "
          f"contact_red_on {mb['c_red_on']:.4f} noncontact_red_learned {mb['nc_red_learned']:.4f} auroc {mb['auroc']:.4f}", flush=True)
    result = {'args': vars(args), 'best_epoch': best_epoch, 'history': history}
    if args.test_data:
        model.load_state_dict(torch.load(out_dir / 'weights_best.pt', map_location=device, weights_only=True))
        trows, toff, tlen = rows_of(args.test_data)
        te_starts = torch.from_numpy(clip_starts(toff, tlen, range(len(tlen)), FS)).to(device)
        mt = evaluate(model, trows, te_starts, d_c)
        result['test'] = mt
        print(f'test ({args.test_data}, {len(te_starts)} clips, best epoch {best_epoch}): {fmt(mt)}', flush=True)
        print(f"RESULT_TEST {args.out} best_epoch {best_epoch} contact_red_learned {mt['c_red_learned']:.4f} "
              f"contact_red_on {mt['c_red_on']:.4f} noncontact_red_learned {mt['nc_red_learned']:.4f} auroc {mt['auroc']:.4f}", flush=True)
    json.dump(result, open(out_dir / 'metrics.json', 'w'), indent=1)
    print('TRAIN_INT DONE', flush=True)


if __name__ == '__main__':
    main()
