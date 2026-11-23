"""E56: 화면에서 obstacle 상태를 읽는 탐지기와 obstacle 궤적 예측기.

정답 obstacle 입력(iawm.obs_oracle)은 planning 중 env에서 obstacle 상태를 직접 읽는다. 여기서는 같은 값을
화면만으로 만든다. 학습 때만 data의 obstacle label(obstacle_pos, obstacle_vel)을 쓴다.

- `ObsDetector`: 연속된 두 화면(1 env step 간격)을 받아 obstacle 중심 heatmap, 중심의 sub-pixel offset,
  속도를 낸다(CenterNet 방식). heatmap의 봉우리마다 obstacle 하나를 읽으므로 개수에 무관하고, obstacle이 모두
  같은 모양이라 생기는 순서 문제도 없다.
- `ObsForecaster`: obstacle 하나의 (위치, 속도)에서 5·d step 뒤(d = 1..N_DEPTH)의 (위치, 속도)를 낸다.
  방 경계 반사를 data에서 배운다.
- `est_columns_episode`: data의 한 episode에서 frame마다 탐지 결과를 1 step 간격으로 이어(tracking) slot 순서를
  맞추고, h_int 입력 column [pos, vel, pos_next, mask] (7·K_MAX)을 만든다(scripts/train_int_frozen.py).
- `plan_features`: planning의 history frame 3장(5 step 간격)에서 같은 형식의 입력을 만든다. history 사이의 대응은
  예측기의 1 block 뒤 위치로 맞추고, 미래 frame은 마지막 history 상태를 예측기로 굴린다(scripts/eval.py).

좌표: label과 같은 (x, y) px다. 화면에서 x가 열(column)인지 행(row)인지는 학습 때 파란 픽셀 중심과 label을
비교해 정하고(`xy_is_colrow`), checkpoint에 저장한다.
"""

from __future__ import annotations

import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from iawm.obs_oracle import POS_CENTER, POS_SCALE, VEL_SCALE

STRIDE = 4          # heatmap 한 칸 = 4px (224 -> 56)
IMG = 224
K_MAX = 4           # 한 화면에서 읽는 obstacle 최대 개수 (h_int 입력 slot 수)
N_DEPTH = 6         # 예측기 depth (block 단위). planning horizon 5 + pos_next 1
FRAMESKIP = 5
OBSTACLE_RGB = (0, 90, 255)  # envs/collision_tworoom.py OBSTACLE_COLOR (좌표 방향 확인에만 쓴다)


class _Block(nn.Module):
    def __init__(self, c, d):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=d, dilation=d, bias=False)
        self.b1 = nn.BatchNorm2d(c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(c)

    def forward(self, x):
        return F.relu(x + self.b2(self.c2(F.relu(self.b1(self.c1(x))))))


class ObsDetector(nn.Module):
    """두 화면 (B, H, W, 3) uint8 -> heatmap logit (B, 1, 56, 56), 회귀 (B, 4, 56, 56) = [offset (2), vel/VEL_SCALE (2)].

    offset과 vel의 두 성분은 label 좌표 (x, y) 순서다.
    """

    def __init__(self, width: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(6, 32, 5, stride=2, padding=2, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, width, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU(),
        )
        self.body = nn.Sequential(*[_Block(width, d) for d in (1, 2, 4, 8, 1, 2)])
        self.hm = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.ReLU(), nn.Conv2d(width, 1, 1))
        self.reg = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.ReLU(), nn.Conv2d(width, 4, 1))
        nn.init.constant_(self.hm[-1].bias, -2.19)  # CenterNet: 처음 sigmoid 0.1
        self.register_buffer('xy_is_colrow', torch.tensor(1))

    def forward(self, prev: torch.Tensor, cur: torch.Tensor):
        x = torch.cat([prev, cur], dim=-1).permute(0, 3, 1, 2).float() / 255.0
        f = self.body(self.stem(x))
        return self.hm(f), self.reg(f)

    # ---- label 좌표 <-> heatmap 격자 ----
    def grid_rc(self, p: torch.Tensor) -> torch.Tensor:
        """label (..., 2) (x, y) px -> 격자 연속 좌표 (..., 2) (row, col)."""
        q = p / STRIDE
        return q[..., [1, 0]] if int(self.xy_is_colrow) else q

    def label_xy(self, rc: torch.Tensor) -> torch.Tensor:
        """격자 (..., 2) (row, col) -> label 축 순서 (x, y)."""
        return rc[..., [1, 0]] if int(self.xy_is_colrow) else rc

    @torch.no_grad()
    def detect(self, prev, cur, thr: float = 0.3, k: int = K_MAX):
        """반환: pos (B, k, 2) px, vel (B, k, 2) px/step, mask (B, k) bool (점수 순)."""
        hm, reg = self.forward(prev, cur)
        h = torch.sigmoid(hm.float())
        peak = (h == F.max_pool2d(h, 3, stride=1, padding=1)) & (h > thr)
        B, _, G, W = h.shape
        score = torch.where(peak, h, torch.zeros_like(h)).view(B, -1)
        s, idx = score.topk(k, dim=1)
        r, c = idx // W, idx % W
        reg = reg.float().view(B, 4, -1)
        g = torch.gather(reg, 2, idx[:, None].expand(-1, 4, -1)).permute(0, 2, 1)   # (B, k, 4)
        cell_xy = self.label_xy(torch.stack([r, c], -1).float())
        pos = (cell_xy + g[..., :2]) * STRIDE
        vel = g[..., 2:] * VEL_SCALE
        return pos, vel, s > thr


def detector_targets(det: ObsDetector, pos: torch.Tensor, vel: torch.Tensor, G: int = IMG // STRIDE, sigma: float = 1.0):
    """pos, vel (B, K, 2) px -> heatmap (B, 1, G, G), 중심 3x3 칸의 회귀 target과 mask."""
    B, K, _ = pos.shape
    dev = pos.device
    rc = det.grid_rc(pos)                                            # (B, K, 2) 연속 격자 좌표
    yy = torch.arange(G, device=dev).float()
    d2 = (yy[None, None, :, None] - rc[..., 0:1, None]) ** 2 + (yy[None, None, None, :] - rc[..., 1:2, None]) ** 2
    hm = torch.exp(-d2 / (2 * sigma ** 2)).amax(1, keepdim=True)     # (B, 1, G, G)
    cell = rc.floor().long()
    peak = torch.zeros(B, 1, G, G, device=dev)
    tgt = torch.zeros(B, 4, G, G, device=dev)
    msk = torch.zeros(B, 1, G, G, device=dev)
    b = torch.arange(B, device=dev)
    for k in range(K):
        r0, c0 = cell[:, k, 0].clamp(0, G - 1), cell[:, k, 1].clamp(0, G - 1)
        peak[b, 0, r0, c0] = 1.0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                r, c = (r0 + dr).clamp(0, G - 1), (c0 + dc).clamp(0, G - 1)
                off_rc = rc[:, k] - torch.stack([r, c], -1).float()   # 그 칸 기준 offset (row, col)
                tgt[b, 0:2, r, c] = det.label_xy(off_rc)
                tgt[b, 2:4, r, c] = vel[:, k] / VEL_SCALE
                msk[b, 0, r, c] = 1.0
    hm = torch.maximum(hm, peak)
    return hm, tgt, msk


def detector_loss(hm_logit, reg, hm_t, reg_t, msk):
    """CenterNet focal loss (heatmap) + L1 (중심 3x3 칸의 offset과 속도)."""
    p = torch.sigmoid(hm_logit.float()).clamp(1e-4, 1 - 1e-4)
    pos = hm_t.eq(1).float()
    neg = 1 - pos
    lp = -(torch.log(p) * (1 - p) ** 2 * pos).sum()
    ln = -(torch.log(1 - p) * p ** 2 * (1 - hm_t) ** 4 * neg).sum()
    n = pos.sum().clamp(min=1)
    l_hm = (lp + ln) / n
    l_off = (torch.abs(reg[:, :2].float() - reg_t[:, :2]) * msk).sum() / msk.sum().clamp(min=1)
    l_vel = (torch.abs(reg[:, 2:].float() - reg_t[:, 2:]) * msk).sum() / msk.sum().clamp(min=1)
    return l_hm + l_off + l_vel, {'hm': l_hm.item(), 'off': l_off.item(), 'vel': l_vel.item()}


class ObsForecaster(nn.Module):
    """obstacle 하나의 상태 (..., 4) = [x, y, vx, vy] (px, px/step) -> (..., N_DEPTH, 4) 5·d step 뒤 상태."""

    def __init__(self, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 4 * N_DEPTH))

    @staticmethod
    def norm(s):
        return torch.cat([(s[..., :2] - POS_CENTER) / POS_SCALE, s[..., 2:] / VEL_SCALE], -1)

    @staticmethod
    def denorm(s):
        return torch.cat([s[..., :2] * POS_SCALE + POS_CENTER, s[..., 2:] * VEL_SCALE], -1)

    def forward(self, s):
        out = self.net(self.norm(s.float())).view(*s.shape[:-1], N_DEPTH, 4)
        return self.denorm(out)


def encode_slots(pos, vel, nxt, mask) -> np.ndarray:
    """slot별 (..., K_MAX, 2) 값과 mask (..., K_MAX) -> (..., 7·K_MAX) [pos, vel, pos_next, mask] (obs_oracle과 같은 scale).

    없는 slot은 0이다.
    """
    m = mask[..., None].astype(np.float32)
    f = lambda a, c, s: ((a - c) / s * m).reshape(*a.shape[:-2], -1)
    return np.concatenate([f(pos, POS_CENTER, POS_SCALE), f(vel, 0.0, VEL_SCALE), f(nxt, POS_CENTER, POS_SCALE),
                           mask.astype(np.float32)], -1).astype(np.float32)


def _assign(prev_xy, cur_xy, max_dist):
    """prev (P, 2)와 cur (C, 2)의 최소 거리 대응. 반환: [(p, c), ...] (거리 max_dist 이하만)."""
    if len(prev_xy) == 0 or len(cur_xy) == 0:
        return []
    d = np.linalg.norm(prev_xy[:, None] - cur_xy[None], axis=-1)
    try:
        from scipy.optimize import linear_sum_assignment
        pi, ci = linear_sum_assignment(d)
    except ImportError:  # K_MAX가 작아 전부 시도해도 된다
        best, pi, ci = None, [], []
        P, C = d.shape
        if P <= C:
            for perm in itertools.permutations(range(C), P):
                cost = d[range(P), perm].sum()
                if best is None or cost < best:
                    best, pi, ci = cost, list(range(P)), list(perm)
        else:
            for perm in itertools.permutations(range(P), C):
                cost = d[perm, range(C)].sum()
                if best is None or cost < best:
                    best, pi, ci = cost, list(perm), list(range(C))
    return [(int(p), int(c)) for p, c in zip(pi, ci) if d[p, c] <= max_dist]


def track_episode(pos, vel, mask, max_dist: float = 20.0):
    """한 episode의 frame별 탐지 (T, K_MAX, 2)·mask (T, K_MAX)를 1 step 간격으로 이어 slot 순서를 맞춘다.

    이전 frame slot의 다음 위치(pos + vel)와 이번 탐지를 대응시킨다. 대응이 없는 탐지는 빈 slot에 넣는다.
    반환: 같은 모양의 slot 정렬 결과.
    """
    T = len(pos)
    sp, sv = np.zeros_like(pos), np.zeros_like(vel)
    sm = np.zeros(mask.shape, bool)
    for t in range(T):
        det = np.nonzero(mask[t])[0]
        if t > 0:
            live = np.nonzero(sm[t - 1])[0]
            pred = sp[t - 1, live] + sv[t - 1, live]
            pairs = _assign(pred, pos[t, det], max_dist)
        else:
            live, pairs = np.array([], int), []
        used_d = set()
        for p, c in pairs:
            s = live[p]
            sp[t, s], sv[t, s], sm[t, s] = pos[t, det[c]], vel[t, det[c]], True
            used_d.add(c)
        # 빈 slot: 직전 frame에도 비어 있던 slot을 먼저 쓴다(방금 놓친 obstacle의 slot은 나중에)
        empty = [s for s in range(pos.shape[1]) if not sm[t, s]]
        free = [s for s in empty if not (t > 0 and sm[t - 1, s])] + [s for s in empty if t > 0 and sm[t - 1, s]]
        for c in range(len(det)):
            if c in used_d or not free:
                continue
            s = free.pop(0)
            sp[t, s], sv[t, s], sm[t, s] = pos[t, det[c]], vel[t, det[c]], True
    return sp, sv, sm


@torch.no_grad()
def forecast_np(fc: ObsForecaster, pos, vel, device) -> np.ndarray:
    """(..., 2), (..., 2) numpy -> (..., N_DEPTH, 4) numpy."""
    s = torch.from_numpy(np.concatenate([pos, vel], -1).astype(np.float32)).to(device)
    return fc(s).cpu().numpy()


def plan_features(det_hist, fc: ObsForecaster, horizon: int, device):
    """planning 입력 (H + horizon - 1, 7·K_MAX).

    det_hist: history frame H장(5 step 간격, 마지막이 지금)의 탐지 [(pos (K_MAX, 2), vel, mask), ...].
    slot 순서는 마지막 frame의 탐지 순서다. 앞 frame은 그 frame 상태를 예측기로 1 block 굴린 위치를 다음 frame의
    slot과 대응시켜 맞춘다. 미래 frame d(1..horizon-1)의 상태는 마지막 history 상태를 d block 굴린 값이다.
    pos_next는 각 frame 상태를 1 block 굴린 위치다(학습 column과 같은 정의).
    """
    H = len(det_hist)
    K = det_hist[-1][0].shape[0]
    P = np.zeros((H, K, 2), np.float32)
    V = np.zeros((H, K, 2), np.float32)
    M = np.zeros((H, K), bool)
    P[-1], V[-1], M[-1] = det_hist[-1]
    for h in range(H - 2, -1, -1):  # 뒤에서 앞으로 맞춘다
        pos, vel, msk = det_hist[h]
        idx = np.nonzero(msk)[0]
        live = np.nonzero(M[h + 1])[0]
        if len(idx) and len(live):
            nxt = forecast_np(fc, pos[idx], vel[idx], device)[:, 0, :2]
            for p, c in _assign(nxt, P[h + 1, live], 30.0):
                s = live[c]
                P[h, s], V[h, s], M[h, s] = pos[idx[p]], vel[idx[p]], True
    f_hist = forecast_np(fc, P, V, device)                           # (H, K, N_DEPTH, 4)
    nxt_hist = f_hist[:, :, 0, :2]
    last = f_hist[-1]                                                # (K, N_DEPTH, 4)
    fut_p = np.stack([last[:, d, :2] for d in range(horizon - 1)])  # (horizon-1, K, 2)
    fut_v = np.stack([last[:, d, 2:] for d in range(horizon - 1)])
    fut_n = np.stack([last[:, d + 1, :2] for d in range(horizon - 1)])
    fut_m = np.repeat(M[-1][None], horizon - 1, 0)
    pos = np.concatenate([P, fut_p]); vel = np.concatenate([V, fut_v])
    nxt = np.concatenate([nxt_hist, fut_n]); msk = np.concatenate([M, fut_m])
    return encode_slots(pos, vel, nxt, msk)


def match_error(pos, vel, mask, true_pos, true_vel):
    """한 화면의 탐지와 정답 (K, 2)의 대응 오차. 반환: (pos 오차 list, vel 오차 list, 놓친 수, 잘못 찾은 수)."""
    det = np.nonzero(mask)[0]
    pairs = _assign(true_pos, pos[det], 15.0)
    pe = [float(np.linalg.norm(true_pos[t] - pos[det[c]])) for t, c in pairs]
    ve = [float(np.linalg.norm(true_vel[t] - vel[det[c]])) for t, c in pairs]
    return pe, ve, len(true_pos) - len(pairs), len(det) - len(pairs)


def blue_centroid_check(pixels: np.ndarray, pos: np.ndarray, radius: float = 22.0):
    """좌표 방향 확인: 다른 obstacle과 떨어진 obstacle의 파란 픽셀 중심과 label을 (x=열)·(x=행) 두 해석으로 비교한다.

    pixels (N, H, W, 3) uint8, pos (N, K, 2). 반환: (colrow 오차 평균, rowcol 오차 평균, 쓴 obstacle 수).
    """
    errs = {True: [], False: []}
    tgt = np.array(OBSTACLE_RGB, np.float32)
    for img, ps in zip(pixels, pos):
        blue = np.linalg.norm(img.astype(np.float32) - tgt, axis=-1) < 40
        rr, cc = np.nonzero(blue)
        for k, p in enumerate(ps):
            if any(np.linalg.norm(p - q) < 3 * radius for j, q in enumerate(ps) if j != k):
                continue
            for colrow in (True, False):
                x, y = (cc, rr) if colrow else (rr, cc)
                sel = (x - p[0]) ** 2 + (y - p[1]) ** 2 < (radius * 1.3) ** 2
                if sel.sum() < 0.6 * np.pi * radius ** 2:
                    continue
                errs[colrow].append(np.hypot(x[sel].mean() - p[0], y[sel].mean() - p[1]))
    cr = float(np.mean(errs[True])) if errs[True] else float('inf')
    rc = float(np.mean(errs[False])) if errs[False] else float('inf')
    return cr, rc, len(errs[True]) + len(errs[False])


def save(path, det: ObsDetector, fc: ObsForecaster, meta: dict):
    torch.save({'det': det.state_dict(), 'fc': fc.state_dict(), 'meta': meta}, path)


def load(path, device='cpu'):
    ck = torch.load(path, map_location=device, weights_only=False)
    det, fc = ObsDetector(**ck['meta'].get('det_kwargs', {})), ObsForecaster()
    det.load_state_dict(ck['det'])
    fc.load_state_dict(ck['fc'])
    return det.to(device).eval(), fc.to(device).eval(), ck['meta']
