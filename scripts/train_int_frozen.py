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
  --oracle-parts pos|posvel (E45 진단): 정답 상태 중 현재 위치만, 또는 위치와 속도만 넣는다(기본 full = 다음 위치까지).
- --oracle forecast (E45): action 없이 다음 block latent를 예측하는 forecaster(iawm.forecaster)를 먼저 학습하고
  (label 없음, --fc-steps), 그 예측을 h_int 입력에 더한다. --fc-steps 0이면 학습하지 않은 forecaster(= 마지막 frame을
  그대로 내는 persistence)로 같은 크기의 대조군을 만든다. forecaster의 예측은 depth 1~5(실제 frame에서 몇 step 굴렸는가)
  별로 미리 계산해 두고, 학습 때 planning과 같은 depth 분포로 고른다(아래 forward 참고). 검증과 test의 주 지표는 depth 1
  (정답 obstacle 입력의 수치와 같은 조건)이고, depth를 planning 마지막 쪽처럼 깊게 한 수치(RESULT_TEST_DEEP)도 남긴다.
  --fc-anchor (E46): forecaster 예측 뒤에 "마지막 실제 frame" latent(anchor_columns)를 이어 붙여 입력을 2배로 한다.
  --fc-data (E48): forecaster를 h_int 학습 data(--data)가 아닌 따로 수집한 data로 학습한다(cross-fit). 같은 data로 학습하면
  h_int가 학습 때 받는 forecaster 예측이 처음 보는 data에서보다 정확해져 그 입력을 과하게 믿을 수 있다(E47의 가설).
  --fc-motion (E50): 마지막 실제 frame과 그 1 env step 전 frame의 latent 차이(motion_columns)를 이어 붙인다. block 간격
  frame으로는 obstacle 2개의 대응이 모호해 속도를 알기 어렵다. 1 step 차이는 label 없이 속도 단서를 준다.
  --fc-depth-tag (E47): 맨 뒤에 그 feature의 column(depth) one-hot(iawm.forecaster.N_TAGS)을 붙인다. planning에서는
  Forecaster.plan_features가 같은 index(tag_index)를 붙인다.
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
import os
import re
import sys
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT), str(ROOT / 'scripts')]

import numpy as np
import torch

from add_contact_labels import contact_next_labels
from analyze_gate import action_stats, episode_offsets, to_model_pixels
from iawm.forecaster import N_TAGS, TAG_DEPTH

INT_RE = re.compile(r'^predictor\.(h_int|int_in|int_out|gate)(\.|$)')      # h_int 단계에서 학습하는 부분
NEW_RE = re.compile(r'^predictor\.(h_int|int_in|int_out|gate|forecaster)(\.|$)')  # base에 없는 부분 (int_only에 남긴다)
FS, NUM_STEPS, HIST = 5, 4, 3  # frameskip, clip 길이(history + 1), history. 학습 config와 같다
FC_DEPTH = 5  # forecaster feature를 미리 계산해 두는 최대 depth (planning horizon 5 block)
FC_TRUNC = (FC_DEPTH, FC_DEPTH + 1)  # 앞 frame이 잘린 depth 1 column: 실제 frame 1개만 / 2개만 본 것 (아래 forecast_columns)
assert N_TAGS == FC_DEPTH + 2 and TAG_DEPTH == FC_DEPTH, 'iawm.forecaster의 tag 정의가 column index와 같아야 한다'
OBS_PARTS = {'pos': 2, 'posvel': 4, 'full': 6}  # obs_oracle 한 obstacle의 앞쪽 몇 차원을 쓰는가 [pos, vel, pos_next]


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
        p.update(int_oracle_dim=OBS_PARTS[args.oracle_parts] * args.num_obstacles, int_oracle_key='obs_oracle')
        if args.per_object:  # E53
            assert args.oracle_parts == 'full'
            p.update(int_per_object=args.num_obstacles)
    if args.oracle == 'est':  # E56: 탐지기가 화면에서 읽은 obstacle 상태 (slot K_MAX개 + mask), obstacle별 h_int
        from iawm.obs_detector import K_MAX
        p.update(int_oracle_dim=7 * K_MAX, int_oracle_key='obs_est', int_per_object=K_MAX, int_obj_mask=True)
    if args.oracle == 'forecast':
        d = p['dim_c'] + p['dim_u']
        p.update(int_oracle_dim=d * (2 if args.fc_anchor else 1) + (d if args.fc_motion else 0) + (N_TAGS if args.fc_depth_tag else 0),
                 int_oracle_key='forecast',
                 forecaster_cfg={'dim': d, 'history': HIST, 'hidden': args.fc_hidden, 'depth': args.fc_depth,
                                 'anchor': bool(args.fc_anchor), 'depth_tag': bool(args.fc_depth_tag),
                                 'motion': bool(args.fc_motion)})
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


def obs_part_columns(orc: np.ndarray, part: str, k: int) -> np.ndarray:
    """obs_oracle column (N, 6K) = [pos (2K), vel (2K), pos_next (2K)]에서 앞쪽 성분만 남긴다 (E45 진단)."""
    return orc[:, :OBS_PARTS[part] * k]


@torch.no_grad()
def est_columns(f, detector: str, device, cache: Path | None, name: str, chunk=2000):
    """E56: file의 모든 row에 대한 탐지기 입력 column (N, 7·K_MAX). cache가 있으면 읽는다.

    row j는 (frame j-1, frame j)로 탐지한다. episode 첫 row는 앞 frame이 없어 (frame j+1, frame j)로 탐지하고
    속도 부호를 뒤집는다. episode마다 1 step 간격으로 이어 slot 순서를 맞추고(track_episode), pos_next는 예측기의
    1 block 뒤 위치다(planning의 iawm.obs_detector.plan_features와 같은 정의). 정답 label과의 오차를 RESULT_EST로 남긴다.
    """
    from iawm import obs_detector as od
    if cache is not None and cache.exists():
        col = np.load(cache)
        if len(col) == len(f['pixels']):
            print(f'est columns: cache {cache} ({col.shape})', flush=True)
            return col
    det, fc, meta = od.load(Path(os.environ.get('STABLEWM_HOME', Path.home() / '.stable-wm')) / 'checkpoints' / detector, device)
    off, ep_len = episode_offsets(f)
    n = len(f['pixels'])
    first = np.zeros(n, bool)
    first[off] = True
    P = np.zeros((n, od.K_MAX, 2), np.float32)
    V = np.zeros((n, od.K_MAX, 2), np.float32)
    M = np.zeros((n, od.K_MAX), bool)
    t0 = time.time()
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        a, b = max(0, lo - 1), min(n, hi + 1)
        px = torch.from_numpy(f['pixels'][a:b]).to(device)
        j = np.arange(lo, hi)
        prev = np.where(first[j], j + 1, j - 1)
        for s in range(0, len(j), 512):
            jj, pp = j[s:s + 512], prev[s:s + 512]
            pos, vel, msk = det.detect(px[pp - a], px[jj - a])
            vel = torch.where(torch.from_numpy(first[jj]).to(device)[:, None, None], -vel, vel)
            P[jj], V[jj], M[jj] = pos.cpu().numpy(), vel.cpu().numpy(), msk.cpu().numpy()
        if (lo // chunk) % 50 == 0:
            print(f'est columns: {hi}/{n} rows ({time.time() - t0:.0f}s)', flush=True)
    for o, L in zip(off, ep_len):
        P[o:o + L], V[o:o + L], M[o:o + L] = od.track_episode(P[o:o + L], V[o:o + L], M[o:o + L])
    nxt = np.zeros_like(P)
    for lo in range(0, n, 65536):
        nxt[lo:lo + 65536] = od.forecast_np(fc, P[lo:lo + 65536], V[lo:lo + 65536], device)[..., 0, :2]
    col = od.encode_slots(P, V, nxt, M)
    # 정답과의 오차 (진단)
    tp = np.nan_to_num(f['obstacle_pos'][:].astype(np.float32)).reshape(n, -1, 2)
    tv = np.nan_to_num(f['obstacle_vel'][:].astype(np.float32)).reshape(n, -1, 2)
    pe, ve, miss, fp = [], [], 0, 0
    for j in np.random.default_rng(0).choice(n, min(n, 20000), replace=False):
        a_, b_, m_, f_ = od.match_error(P[j], V[j], M[j], tp[j], tv[j])
        pe += a_; ve += b_; miss += m_; fp += f_
    k = tp.shape[1]
    print(f'RESULT_EST {name} pos_mae {np.mean(pe):.3f} vel_mae {np.mean(ve):.3f} vel_p90 {np.quantile(ve, .9):.3f} '
          f'miss {miss / (20000 * k):.4f} fp {fp / (20000 * k):.4f} slots_used {M.sum(1).mean():.2f} ({time.time() - t0:.0f}s)', flush=True)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_name(f'{cache.stem}.tmp{int(time.time() * 1e6)}.npy')
        np.save(tmp, col)
        tmp.replace(cache)
    return col


def load_rows(model, path: Path, device, cache: Path | None, oracle: str, oracle_parts: str = 'full', k: int = 1,
              noise: float = 0.0, noise_seed: int = 0, detector: str | None = None):
    """학습에 쓰는 row별 tensor를 GPU에 올린다. 정의는 analyze_gate.iter_clips, add_contact_labels와 같다."""
    import h5py
    import hdf5plugin  # noqa: F401

    from iawm import obs_oracle
    from multistep_px import positions

    with h5py.File(path, 'r') as f:
        off, ep_len = episode_offsets(f)
        z = frame_latents(model, f, device, cache)
        mean, std = action_stats(f)
        act = np.nan_to_num((f['action'][:].astype(np.float32) - mean) / std, nan=0.0).astype(np.float32)
        contact = f['contact'][:].reshape(len(f['contact']), -1)[:, 0]
        cn = contact_next_labels(contact, off, ep_len, FS)  # row j: frame j -> j+5 사이 contact
        orc = obs_part_columns(obs_oracle.dataset_column(f, FS), oracle_parts, k) if oracle == 'obs' else None
        if orc is not None and noise > 0:  # E52: 속도 추정 오차 (row마다 한 번 뽑아 고정한다)
            assert oracle_parts == 'full', '--oracle-noise는 full 입력에서만 쓴다 (planning도 full만 넣는다)'
            orc = obs_oracle.vel_noise(orc, noise, k, np.random.default_rng(noise_seed), FS)
            print(f'oracle velocity noise sigma {noise} px/step on {path.name} (seed {noise_seed})', flush=True)
        if oracle == 'est':  # E56
            orc = est_columns(f, detector, device, None if cache is None else cache.with_name(
                f'est__{Path(detector).stem}__{path.stem}.npy'), path.name)
        pos = positions(f) if oracle == 'forecast' else None  # forecaster px 진단에만 쓴다(학습에는 쓰지 않는다)
    rows = {'z': torch.from_numpy(z).to(device), 'act': torch.from_numpy(act).to(device),
            'cn': torch.from_numpy(cn).to(device)}
    if orc is not None:
        rows['orc'] = torch.from_numpy(np.ascontiguousarray(orc)).to(device)
    if pos is not None:
        rows['pos'] = pos
    row_off = np.repeat(off, ep_len).astype(np.int64)  # row별 episode 시작 row
    rows['row_off'] = torch.from_numpy(row_off).to(device)
    return rows, off, ep_len


# ---------------------------------------------------------------- forecaster (E45)
def hist_rows(r, row_off, n=HIST):
    """frame row r (B,)의 history row (B, n): r-(n-1)·FS .. r. episode 시작 전은 첫 frame을 반복한다."""
    k = torch.arange(n - 1, -1, -1, device=r.device) * FS
    return torch.maximum(r[:, None] - k[None], row_off[r][:, None])


def train_forecaster(fc, rows, off, ep_len, episodes, steps, device, batch=2048, lr=1e-3, k_max=FC_DEPTH):
    """label 없이 학습: 실제 history에서 1..k_max step 굴린 예측이 실제 미래 frame latent를 맞히게 한다."""
    z, row_off = rows['z'], rows['row_off']
    span = FS * k_max
    starts = np.concatenate([off[e] + np.arange(0, ep_len[e] - span) for e in episodes if ep_len[e] > span])
    starts = torch.from_numpy(starts.astype(np.int64)).to(device)
    fc.requires_grad_(True)  # main()에서 base와 함께 꺼 두었다
    opt = torch.optim.AdamW(fc.net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    fc.train()
    t0 = time.time()
    for it in range(steps):
        r = starts[torch.randint(0, len(starts), (batch,), device=device)]
        hist = z[hist_rows(r, row_off)]
        pred = fc.roll(hist, k_max)                                                    # (B, K, D)
        tgt = z[r[:, None] + FS * torch.arange(1, k_max + 1, device=device)[None]]      # (B, K, D)
        loss = ((pred - tgt) / fc.zs).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(fc.net.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % 2000 == 0 or it == steps - 1:
            print(f'forecaster step {it} loss {loss.item():.4f} ({time.time() - t0:.0f}s)', flush=True)
    fc.eval()


@torch.no_grad()
def forecast_columns(fc, rows, batch=32768):
    """row r마다 forecaster가 예측한 frame r+FS의 latent. 반환 (FC_DEPTH + 2, N, D) float16.

    - index d-1 (d = 1..FC_DEPTH): frame r-FS·(d-1)까지의 실제 frame에서 d step 굴린 값(planning에서 history 밖 frame의
      feature와 같은 정의). episode 시작이 가까워 그만큼 과거가 없으면 가능한 가장 큰 depth를 쓴다.
    - index FC_TRUNC[0], FC_TRUNC[1]: planning의 history 첫 두 frame에 해당하는 값. planning에는 history 3 frame보다
      앞의 frame이 없어서 첫 frame을 반복한다. 실제 frame을 1개만([r, r, r]), 2개만([r-FS, r-FS, r]) 본 depth 1 예측이다.
    """
    assert HIST == 3, 'FC_TRUNC는 history 3 frame 기준이다'
    z, row_off = rows['z'], rows['row_off']
    n, d = z.shape
    out = torch.empty(FC_DEPTH + 2, n, d, dtype=torch.float16, device=z.device)
    for i in range(0, n, batch):
        r = torch.arange(i, min(i + batch, n), device=z.device)
        back = (r - row_off[r]) // FS                                                  # 실제 과거 block 수
        for depth in range(1, FC_DEPTH + 1):
            de = torch.clamp(back + 1, max=depth)                                       # 실제 depth (B,)
            e = r - FS * (de - 1)
            roll = fc.roll(z[hist_rows(e, row_off)], depth)                             # (B, depth, D)
            out[depth - 1, i:i + len(r)] = roll[torch.arange(len(r), device=z.device), de - 1].half()
        for col, n_real in zip(FC_TRUNC, (1, 2)):
            k = torch.tensor([min(2, n_real - 1), min(1, n_real - 1), 0], device=z.device) * FS
            h = torch.maximum(r[:, None] - k[None], row_off[r][:, None])
            out[col, i:i + len(r)] = fc.step(z[h]).half()
    return out


@torch.no_grad()
def forecast_gap(rows, off, ep_len, tr_eps, va_eps):
    """forecast column별 예측 오차를 h_int 학습 episode와 검증 episode에서 따로 잰다 (E48 진단).

    column d-1(depth d)은 frame r+FS를 예측한다. 오차는 frame latent 표준편차로 나눈 MSE다. 학습 episode 쪽이 확연히
    작으면 h_int는 학습 때 처음 보는 data보다 정확한 입력을 받은 것이다. 반환 [(depth, train, val), ...].
    """
    z, fcc = rows['z'], rows['fc']
    zs = z.float().std(0).clamp_min(1e-6)

    def err(col, eps):
        r = np.concatenate([off[e] + np.arange(0, ep_len[e] - FS) for e in eps if ep_len[e] > FS])
        r = torch.from_numpy(r.astype(np.int64)).to(z.device)
        tot = 0.0
        for i in range(0, len(r), 65536):
            b = r[i:i + 65536]
            tot += (((fcc[col, b].float() - z[b + FS].float()) / zs).pow(2).mean(-1)).sum().item()
        return tot / len(r)
    res = [(d, err(d - 1, tr_eps), err(d - 1, va_eps)) for d in range(1, FC_DEPTH + 1)]
    print('forecast error train/val by depth: ' + ' '.join(f'd{d} {a:.4f}/{b:.4f}' for d, a, b in res), flush=True)
    return res


@torch.no_grad()
def anchor_columns(rows):
    """forecast_columns와 같은 index의 "마지막 실제 frame" latent (E46). 반환 (FC_DEPTH + 2, N, D) float16.

    depth d column은 frame r-FS·(d-1)(episode 시작에서 잘린 실제 depth 기준), 잘린 column 두 개와 depth 1은 frame r이다.
    학습하지 않은 forecaster(persistence)의 forecast_columns와 정확히 같은 값이다(E45 대조군 fc0의 입력).
    """
    z, row_off = rows['z'], rows['row_off']
    n = z.shape[0]
    r = torch.arange(n, device=z.device)
    back = (r - row_off) // FS
    out = torch.empty(FC_DEPTH + 2, n, z.shape[1], dtype=torch.float16, device=z.device)
    for depth in range(1, FC_DEPTH + 1):
        de = torch.clamp(back + 1, max=depth)
        out[depth - 1] = z[r - FS * (de - 1)].half()
    out[FC_TRUNC[0]] = z.half()
    out[FC_TRUNC[1]] = z.half()
    return out


@torch.no_grad()
def motion_columns(rows):
    """anchor_columns와 같은 index의 "마지막 실제 frame - 그 1 env step 전 frame" latent 차이 (E50). 반환 (FC_DEPTH + 2, N, D) float16.

    anchor row a마다 z[a] - z[a - 1]이다. episode 첫 frame이면 앞 frame이 없으므로 z[a + 1] - z[a]를 쓴다.
    planning에서는 history frame마다 1 step 전 frame(첫 계획이면 env.render_past)의 latent와의 차이다(iawm.forecaster).
    """
    z, row_off = rows['z'], rows['row_off']
    n = z.shape[0]
    r = torch.arange(n, device=z.device)
    first = r == row_off
    nxt = torch.clamp(r + 1, max=n - 1)
    dz = torch.where(first[:, None], z[nxt] - z, z - z[torch.clamp(r - 1, min=0)]).half()   # (N, D): row r의 1 step 차이
    back = (r - row_off) // FS
    out = torch.empty(FC_DEPTH + 2, n, z.shape[1], dtype=torch.float16, device=z.device)
    for depth in range(1, FC_DEPTH + 1):
        de = torch.clamp(back + 1, max=depth)
        out[depth - 1] = dz[r - FS * (de - 1)]
    out[FC_TRUNC[0]] = dz
    out[FC_TRUNC[1]] = dz
    return out


@torch.no_grad()
def forecaster_px(fc, rows, off, ep_len, tr_eps, te_eps, device, d_c, ks=(1, 3, 5), n_fit=150000, steps=4000):
    """진단: 미래 obstacle 위치(px)를 얼마나 꺼낼 수 있는가. 위치 label은 이 진단에만 쓴다(모델 학습에는 쓰지 않는다).

    k block 뒤 obstacle 위치를 맞히는 MLP probe(iawm.pos_probe)를 학습 episode에서 맞추고 검증 episode에서 잰다.
    - fc:   forecaster가 k step 굴린 예측 latent (192)에서
    - hist: 실제 history 3 frame latent를 이어 붙인 것 (576)에서. h_int가 이미 보는 정보로 꺼낼 수 있는 양(상한 쪽 참고)
    - now:  지금 frame의 위치 그대로 (persistence, 실제 frame probe)
    forecast latent는 agent 성분이 평균으로 흐려져 실제 frame용 probe로 바로 읽으면 왜곡되므로, fc용 probe를 따로 맞춘다.
    """
    from iawm import pos_probe
    z, row_off = rows['z'], rows['row_off']
    P = torch.from_numpy(rows['pos']).to(device)
    span = FS * max(ks)

    def starts(eps, step):
        return torch.from_numpy(np.concatenate([off[e] + np.arange(0, ep_len[e] - span, step) for e in eps
                                                if ep_len[e] > span]).astype(np.int64)).to(device)

    r_tr, r_te = starts(tr_eps, 1), starts(te_eps, FS)
    r_tr = r_tr[torch.randperm(len(r_tr), generator=torch.Generator().manual_seed(0))[:n_fit].to(device)]

    def feats(r):
        h = z[hist_rows(r, row_off)]
        return h, fc.roll(h, max(ks))

    h_tr, f_tr = feats(r_tr)
    h_te, f_te = feats(r_te)

    def probe_err(Xtr, Ytr, Xte, Yte):
        with torch.enable_grad():
            pr, _ = pos_probe.fit(Xtr.float().cpu().numpy(), Ytr.cpu().numpy(), device, steps=steps)
        return (pr(Xte.float()) - Yte).norm(dim=-1).mean().item()

    with torch.enable_grad():
        now_pr, _ = pos_probe.fit(z[r_tr, :d_c].float().cpu().numpy(), P[r_tr, 2:4].cpu().numpy(), device, steps=steps)
    floor = (now_pr(z[r_te, :d_c].float()) - P[r_te, 2:4]).norm(dim=-1).mean().item()
    res = []
    for k in ks:
        y_tr, y_te = P[r_tr + FS * k, 2:4], P[r_te + FS * k, 2:4]
        e_fc = probe_err(f_tr[:, k - 1], y_tr, f_te[:, k - 1], y_te)
        e_h = probe_err(h_tr.flatten(1), y_tr, h_te.flatten(1), y_te)
        e_now = (now_pr(z[r_te, :d_c].float()) - y_te).norm(dim=-1).mean().item()
        res.append((k, e_fc, e_h, e_now))
    s = ' '.join(f'k{k} {a:.2f}/{b:.2f}/{c:.2f}' for k, a, b, c in res)
    print(f'future obstacle px (forecast/history/persistence): {s} | current-frame probe {floor:.2f}', flush=True)
    return res


def clip_starts(off, ep_len, episodes, stride):
    """clip 시작 row. clip은 frame row s, s+5, s+10, s+15와 action row s..s+19다(학습 loader와 같다)."""
    span = NUM_STEPS * FS
    out = [off[ep] + np.arange(0, ep_len[ep] - span + 1, stride) for ep in episodes if ep_len[ep] >= span]
    return np.concatenate(out).astype(np.int64)


def forward(model, rows, s, call=None):
    """clip 시작 row s (B,)에 대한 예측. 반환: pred, tgt, base, delta, logit, alpha, label (모두 position 0..2).

    forecaster feature(rows['fc'])를 고르는 법 (E45). planning rollout의 t번째 예측(t = 0..5)에서 window 위치 j(0..2)는
    planning frame i = t + j다. history(i = 0, 1, 2)는 실제 frame이고 그보다 앞의 frame은 없다.
    - i = 0, 1: 앞 frame이 잘린 depth 1 값 (FC_TRUNC)
    - i ≥ 2: depth i - 1 (최대 FC_DEPTH). i = 2는 실제 3 frame을 본 depth 1이다.
    call=None이면 학습처럼 t를 sample마다 무작위로 고르고, 숫자면 모든 sample에 그 t를 쓴다.
    call='full'이면 모든 위치에 실제 3 frame을 본 depth 1 값을 쓴다(E45 판정 1의 주 지표. 정답 입력과 같은 조건).
    """
    pr = model.predictor
    fr = s[:, None] + FS * torch.arange(NUM_STEPS, device=s.device)[None]            # (B, 4)
    emb = rows['z'][fr]                                                               # (B, 4, D)
    act = rows['act'][s[:, None] + torch.arange(NUM_STEPS * FS, device=s.device)[None]].reshape(len(s), NUM_STEPS, -1)
    act_emb = model.action_encoder(act)
    if 'orc' in rows:
        pr.oracle = rows['orc'][fr[:, :HIST]]
    if 'fc' in rows:
        if call == 'full':
            col = torch.zeros(len(s), HIST, dtype=torch.long, device=s.device)
        else:
            t = (torch.randint(0, FC_DEPTH + 1, (len(s), 1), device=s.device) if call is None
                 else torch.full((len(s), 1), int(call), device=s.device))
            i = t + torch.arange(HIST, device=s.device)[None]                           # planning frame (B, 3)
            col = torch.where(i == 0, FC_TRUNC[0], torch.where(i == 1, FC_TRUNC[1], torch.clamp(i - 1, 1, FC_DEPTH) - 1))
        pr.oracle = rows['fc'][col, fr[:, :HIST]].float()
        if 'anc' in rows:  # E46: 마지막 실제 frame을 이어 붙인다 (같은 column index)
            pr.oracle = torch.cat([pr.oracle, rows['anc'][col, fr[:, :HIST]].float()], dim=-1)
        if 'mot' in rows:  # E50: 1 env step 차이를 이어 붙인다 (같은 column index)
            pr.oracle = torch.cat([pr.oracle, rows['mot'][col, fr[:, :HIST]].float()], dim=-1)
        if pr.forecaster is not None and pr.forecaster.depth_tag:  # E47: column index one-hot
            pr.oracle = torch.cat([pr.oracle, torch.nn.functional.one_hot(col, N_TAGS).float()], dim=-1)
    pred = model.predict(emb[:, :HIST], act_emb[:, :HIST])
    return {'pred': pred, 'tgt': emb[:, 1:HIST + 1], 'base': pr.last_pred_c_base, 'delta': pr.last_delta,
            'logit': pr.last_logit.squeeze(-1), 'alpha': pr.last_alpha.squeeze(-1), 'lab': rows['cn'][fr[:, :HIST]] > 0.5}


@torch.no_grad()
def evaluate(model, rows, starts, d_c, batch=4096, call='full'):
    """contact / non-contact별 z^c 오차. off = base(CF-JEPA), on = α 1, learned = 학습된 α."""
    model.eval()
    acc = {k: [] for k in ('off', 'on', 'learned', 'alpha', 'lab')}
    for i in range(0, len(starts), batch):
        o = forward(model, rows, starts[i:i + batch], call)
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
    ap.add_argument('--oracle', default='none', choices=['none', 'obs', 'forecast', 'est'])
    ap.add_argument('--detector', default=None, help='E56 --oracle est: 탐지기 checkpoint ($STABLEWM_HOME/checkpoints 기준)')
    ap.add_argument('--oracle-noise', type=float, default=0.0, help='E52: obs 입력의 속도 오차 표준편차 (px/step). planning에는 +oracle_noise로 같은 값을 준다')
    ap.add_argument('--per-object', action='store_true', help='E53: obstacle별로 같은 h_int를 돌려 보정을 더한다 (--oracle obs)')
    ap.add_argument('--oracle-parts', default='full', choices=list(OBS_PARTS), help='E45 진단: obs 입력 중 쓸 성분')
    ap.add_argument('--fc-steps', type=int, default=20000, help='forecaster 학습 step (0 = 학습하지 않은 대조군)')
    ap.add_argument('--fc-hidden', type=int, default=1024)
    ap.add_argument('--fc-depth', type=int, default=3)
    ap.add_argument('--fc-anchor', action='store_true', help='E46: forecaster 예측과 마지막 실제 frame을 함께 넣는다')
    ap.add_argument('--fc-depth-tag', action='store_true', help='E47: forecaster feature에 depth one-hot을 붙인다')
    ap.add_argument('--fc-motion', action='store_true', help='E50: 마지막 실제 frame의 1 env step 차이를 붙인다')
    ap.add_argument('--fc-data', default=None, help='E48: forecaster 학습 data (기본: --data의 학습 episode, 같은 data)')
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
    bad = [k for k in missing if not NEW_RE.match(k)]
    assert not unexpected and not bad, f'base mismatch: unexpected={unexpected[:5]} missing={bad[:5]}'
    n_train = 0
    for name, p in model.named_parameters():
        p.requires_grad_(bool(INT_RE.match(name)))
        n_train += p.numel() if p.requires_grad else 0
    model = model.to(device).eval()
    pr, d_c = model.predictor, model.dim_c
    print(f'base={args.base} loaded {len(sd)} tensors, new={len(missing)}, trainable={n_train / 1e6:.3f}M '
          f'oracle={args.oracle} parts={args.oracle_parts} noise={args.oracle_noise} per_object={args.per_object} detector={args.detector} anchor={args.fc_anchor} motion={args.fc_motion} depth_tag={args.fc_depth_tag} int_input={args.int_input} depth={args.int_depth} '
          f'hidden={args.int_hidden_dim}', flush=True)

    cache_dir = Path(args.cache_dir) if args.cache_dir else home / 'latent_cache'
    tag = args.base.replace('/', '_').replace('.pt', '')

    fc = pr.forecaster
    fc_ready = [False]

    def rows_of(name):
        r, o, l = load_rows(model, home / 'datasets' / name, device, cache_dir / f'{tag}__{Path(name).stem}.npy',
                            args.oracle, args.oracle_parts, args.num_obstacles,
                            args.oracle_noise, args.seed * 1000 + zlib.crc32(name.encode()) % 1000, args.detector)
        if fc is not None and fc_ready[0]:  # forecaster를 고정한 뒤 읽는 test data
            r['fc'] = forecast_columns(fc, r)
            if fc.anchor:
                r['anc'] = anchor_columns(r)
            if fc.motion:
                r['mot'] = motion_columns(r)
        return r, o, l

    rows, off, ep_len = rows_of(args.data)
    n_ep = len(ep_len) if args.max_episodes is None else min(args.max_episodes, len(ep_len))
    n_val = min(args.val_episodes, n_ep // 5)
    tr_starts = torch.from_numpy(clip_starts(off, ep_len, range(0, n_ep - n_val), args.stride)).to(device)
    va_starts = torch.from_numpy(clip_starts(off, ep_len, range(n_ep - n_val, n_ep), FS)).to(device)
    print(f'data={args.data} episodes train {n_ep - n_val} val {n_val} clips train {len(tr_starts)} val {len(va_starts)}', flush=True)
    fc_px = None
    fc_gap = None
    if fc is not None:  # E45: forecaster를 먼저 학습(label 없음)하고 고정한 뒤 feature column을 만든다
        if args.fc_data:  # E48: cross-fit. h_int 학습 data와 겹치지 않는 data의 모든 episode로 학습한다
            assert Path(args.fc_data).name != Path(args.data).name, '--fc-data는 --data와 달라야 한다'
            frows, foff, flen = rows_of(args.fc_data)
            f_eps = range(len(flen))
            print(f'forecaster data={args.fc_data} episodes {len(flen)} (cross-fit)', flush=True)
        else:
            frows, foff, flen, f_eps = rows, off, ep_len, range(0, n_ep - n_val)
        f_rows = torch.from_numpy(np.concatenate([np.arange(foff[e], foff[e] + flen[e]) for e in f_eps])).to(device)
        fc.set_stats(frows['z'][f_rows])
        del f_rows
        if args.fc_steps > 0:
            train_forecaster(fc, frows, foff, flen, f_eps, args.fc_steps, device)
        del frows
        fc.eval().requires_grad_(False)
        fc_px = forecaster_px(fc, rows, off, ep_len, range(0, n_ep - n_val), range(n_ep - n_val, n_ep), device, d_c)
        rows['fc'] = forecast_columns(fc, rows)
        if fc.anchor:
            rows['anc'] = anchor_columns(rows)
        if fc.motion:
            rows['mot'] = motion_columns(rows)
        fc_ready[0] = True
        fc_gap = forecast_gap(rows, off, ep_len, range(0, n_ep - n_val), range(n_ep - n_val, n_ep))
        print(f"forecast columns {tuple(rows['fc'].shape)} ({rows['fc'].numel() * 2 / 1e9:.2f} GB)", flush=True)

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
            torch.save({k: v for k, v in model.state_dict().items() if NEW_RE.match(k)}, out_dir / 'int_only_best.pt')
        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save(model.state_dict(), out_dir / f'weights_epoch_{epoch}.pt')
        print(f"epoch {epoch} ({time.time() - t0:.0f}s) train fwd/aux/bce {m['train_fwd']:.5f}/{m['train_aux']:.5f}/{m['train_bce']:.4f} "
              f"val: {fmt(m)}{flag}", flush=True)
    if best_epoch == 0:  # 한 번도 base보다 좋아지지 않았다: 마지막 epoch를 best로 남겨 뒤 단계가 돌 수 있게 한다
        torch.save(model.state_dict(), out_dir / 'weights_best.pt')
    mb = history[best_epoch]
    print(f"RESULT_VAL {args.out} best_epoch {best_epoch} contact_red_learned {mb['c_red_learned']:.4f} "
          f"contact_red_on {mb['c_red_on']:.4f} noncontact_red_learned {mb['nc_red_learned']:.4f} auroc {mb['auroc']:.4f}", flush=True)
    result = {'args': vars(args), 'best_epoch': best_epoch, 'history': history, 'forecaster_px': fc_px, 'forecast_gap': fc_gap}
    if 'fc' in rows:  # E47: 검증 data의 깊은 구간(planning t = 4) 수치. test를 보지 않고 planning에 쓸 모델을 고를 때 쓴다
        model.load_state_dict(torch.load(out_dir / 'weights_best.pt', map_location=device, weights_only=True))
        mvd = evaluate(model, rows, va_starts, d_c, call=FC_DEPTH - 1)
        result['val_deep'] = mvd
        print(f"RESULT_VAL_DEEP {args.out} contact_red_learned {mvd['c_red_learned']:.4f} "
              f"contact_red_on {mvd['c_red_on']:.4f} noncontact_red_learned {mvd['nc_red_learned']:.4f}", flush=True)
        mv0 = evaluate(model, rows, va_starts, d_c, call=0)  # E50: 검증 data의 첫 계획 시점(planning t = 0)
        result['val_plan0'] = mv0
        print(f"RESULT_VAL_PLAN0 {args.out} contact_red_learned {mv0['c_red_learned']:.4f} "
              f"contact_red_on {mv0['c_red_on']:.4f} noncontact_red_learned {mv0['nc_red_learned']:.4f}", flush=True)
    if fc_gap is not None:
        print('RESULT_FCGAP ' + args.out + ' ' + ' '.join(f'd{d} {a:.4f} {b:.4f}' for d, a, b in fc_gap), flush=True)
    if fc_px is not None:
        print('RESULT_FCPX ' + args.out + ' ' + ' '.join(f'k{k} {a:.3f} {b:.3f} {c:.3f}' for k, a, b, c in fc_px), flush=True)
    if args.test_data:
        model.load_state_dict(torch.load(out_dir / 'weights_best.pt', map_location=device, weights_only=True))
        trows, toff, tlen = rows_of(args.test_data)
        te_starts = torch.from_numpy(clip_starts(toff, tlen, range(len(tlen)), FS)).to(device)
        mt = evaluate(model, trows, te_starts, d_c)
        result['test'] = mt
        print(f'test ({args.test_data}, {len(te_starts)} clips, best epoch {best_epoch}): {fmt(mt)}', flush=True)
        print(f"RESULT_TEST {args.out} best_epoch {best_epoch} contact_red_learned {mt['c_red_learned']:.4f} "
              f"contact_red_on {mt['c_red_on']:.4f} noncontact_red_learned {mt['nc_red_learned']:.4f} auroc {mt['auroc']:.4f}", flush=True)
        if 'fc' in trows:  # planning과 같은 feature: 첫 예측(t = 0, 앞 frame이 잘린 값 포함)과 마지막 쪽(t = 4, depth 3~5)
            for name, call in (('PLAN0', 0), ('DEEP', FC_DEPTH - 1)):
                md = evaluate(model, trows, te_starts, d_c, call=call)
                result[f'test_{name.lower()}'] = md
                print(f"RESULT_TEST_{name} {args.out} contact_red_learned {md['c_red_learned']:.4f} "
                      f"contact_red_on {md['c_red_on']:.4f} noncontact_red_learned {md['nc_red_learned']:.4f}", flush=True)
    json.dump(result, open(out_dir / 'metrics.json', 'w'), indent=1)
    print('TRAIN_INT DONE', flush=True)


if __name__ == '__main__':
    main()
