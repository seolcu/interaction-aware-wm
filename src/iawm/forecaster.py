"""Action 없이 다음 block의 latent를 예측하는 forecaster (E45).

E39에서 h_int에 정답 obstacle 궤적(현재 위치, 속도, 다음 위치)을 주면 contact z^c 오차가 74% 줄었지만, latent만
주면 5.5%였다. E38에서 CF-JEPA의 z^c에는 obstacle 위치가 1.8px로 들어 있지만 속도는 1.7px/step으로 흐리다.
h_int에 빠진 것은 "다음 block에 obstacle이 어디 있을까"다.

Collision env의 obstacle은 action과 agent의 영향을 받지 않고 움직인다. 그래서 과거 frame의 latent만 보고 action
없이 다음 frame의 latent를 맞히도록 학습하면, 줄일 수 있는 오차는 obstacle(과 action과 무관한 다른 것)의 움직임뿐이다.
agent 성분은 action을 모르므로 평균으로 수렴한다. 정답 label 없이 고정된 encoder의 미래 frame latent만 target으로
쓴다(self-supervised). 이 forecaster의 예측을 정답 obstacle 입력 자리(`int_oracle_key='forecast'`)에 넣는다.

- 입력: 같은 frameskip 간격의 과거 latent history개 (전체 z = [z^c, z^u]), 마지막 두 frame의 차이를 함께 넣는다.
- 출력: 다음 block latent (전체 z). 마지막 층을 0으로 초기화해 학습 전에는 마지막 frame을 그대로 낸다(persistence).
  그래서 학습하지 않은 forecaster(대조군 `fc0`)는 "미래를 모르는" 같은 크기의 입력이 된다.
- planning(`plan_features`): history의 실제 latent에서 시작해 자기 예측을 다시 넣어 미래로 굴린다. action을 받지 않으므로
  CEM 후보 300개가 같은 값을 쓰고, 계획마다 한 번만 계산한다.

frame i의 feature는 "frame i까지 보고 예측한 frame i+1의 latent"다. frame i가 history 안이면 실제 frame만 보고(depth 1),
history 밖이면 history 마지막 frame에서 (i - H + 2) step 굴린 값이다. scripts/train_int_frozen.py의 학습 column도 같은
정의를 따른다(depth 1~5).
"""

from __future__ import annotations

import torch
from torch import nn


TAG_DEPTH = 5  # depth_tag의 최대 depth (scripts/train_int_frozen.py FC_DEPTH와 같다)
N_TAGS = TAG_DEPTH + 2  # depth 1..5 + 앞 frame이 잘린 depth 1 두 종류 (train_int_frozen의 column index와 같다)


def tag_index(i: int, history: int) -> int:
    """planning frame i의 feature가 학습 column 중 어느 것에 해당하는가 (train_int_frozen.forward의 col과 같은 정의).

    i < history - 1: 앞 frame이 잘린 depth 1 (TAG_DEPTH + i). 그 뒤: depth min(i - history + 2, TAG_DEPTH)의 index (depth - 1).
    """
    if i < history - 1:
        return TAG_DEPTH + i
    return min(i - history + 2, TAG_DEPTH) - 1


class Forecaster(nn.Module):
    """depth_tag=True (E47): feature 맨 뒤에 "이 feature가 몇 step 굴린 예측인가"의 one-hot(N_TAGS)을 붙인다.

    E46에서 h_int는 forecaster 예측이 정확한 가까운 위치(depth 1)에 맞춰지고, 깊은 위치(depth 3~5)에서는 이득이 11%로
    떨어졌다(실제 frame만 넣은 대조군 22%). h_int 하나가 모든 depth를 구분 없이 맞추기 때문이라고 추정한다. depth를
    알려 주면 h_int가 depth마다 입력을 다르게 쓸 수 있다. parameter는 없다.

    anchor=True (E46): feature 뒤에 "그 frame까지 본 마지막 실제 frame"의 latent를 이어 붙인다(출력 2·dim).

    E45에서 학습하지 않은 forecaster(= 마지막 실제 frame)를 넣은 대조군이 깊은 depth(3~5)에서 contact 오차를 22% 줄여
    학습한 forecaster(11%)보다 나았다. 실제 frame은 굴린 예측과 달리 흐려지지 않으므로, 둘을 같이 주면 가까운 곳은
    예측이, 먼 곳은 실제 frame과의 차이가 맡을 수 있다. anchor는 parameter가 없고, 학습하지 않은 forecaster의 출력과
    정확히 같은 값이다(scripts/train_int_frozen.py anchor_columns).

    motion=True (E50): 그 다음에 "마지막 실제 frame과 그 1 env step 전 frame의 latent 차이"를 붙인다(+dim).
    block 간격(frameskip) frame에서는 obstacle 2개가 15~90px씩 움직여 대응이 모호하지만, 1 step 사이에는 3~18px만
    움직이므로 속도 단서가 분명하다. planning에서는 history frame마다 1 step 전 frame의 latent(prev)를 받아야 한다.
    parameter는 없다(scripts/train_int_frozen.py motion_columns).
    """

    def __init__(self, dim: int = 192, history: int = 3, hidden: int = 1024, depth: int = 3, anchor: bool = False,
                 depth_tag: bool = False, motion: bool = False):
        super().__init__()
        self.dim, self.history, self.anchor, self.depth_tag, self.motion = dim, history, anchor, depth_tag, motion
        in_dim = dim * (history + 1)
        layers: list[nn.Module] = [nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers.append(nn.Linear(hidden, dim))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer('zm', torch.zeros(dim))
        self.register_buffer('zs', torch.ones(dim))

    def set_stats(self, z: torch.Tensor) -> None:
        """학습 data의 latent (N, dim)로 정규화 통계를 정한다."""
        self.zm.copy_(z.float().mean(0))
        self.zs.copy_(z.float().std(0) + 1e-6)

    def step(self, hist: torch.Tensor) -> torch.Tensor:
        """hist (B, history, dim) -> 다음 block latent (B, dim)."""
        x = (hist.float() - self.zm) / self.zs
        feats = torch.cat([x.flatten(1), x[:, -1] - x[:, -2]], dim=-1)
        return (hist[:, -1].float() + self.net(feats) * self.zs).to(hist.dtype)

    def roll(self, hist: torch.Tensor, steps: int) -> torch.Tensor:
        """hist (B, history, dim)에서 steps번 굴린 예측 (B, steps, dim)."""
        out = []
        for _ in range(steps):
            p = self.step(hist)
            out.append(p)
            hist = torch.cat([hist[:, 1:], p[:, None]], dim=1)
        return torch.stack(out, dim=1)

    @torch.no_grad()
    def plan_features(self, hist: torch.Tensor, n_frames: int, prev: torch.Tensor | None = None) -> torch.Tensor:
        """planning용 feature. hist (B, H, dim) 실제 history latent. 반환 (B, n_frames, dim) (anchor면 2·dim, depth_tag면 + N_TAGS).

        feature[i] = frame i까지(history 밖이면 forecaster가 굴린 frame) 보고 예측한 frame i+1.
        history보다 앞의 frame이 필요하면 첫 frame을 반복한다(학습 column과 같다).
        anchor[i] = frame i까지 본 마지막 실제 frame = hist[min(i, H - 1)].
        motion[i] = hist[a] - prev[a], a = min(i, H - 1) (motion일 때. prev (B, H, dim)는 history frame마다 1 env step 전 frame).
        tag[i] = one-hot(tag_index(i, H)) (depth_tag일 때, 학습 column의 index와 같다).
        """
        H = hist.size(1)
        assert not self.motion or (prev is not None and prev.shape == hist.shape), 'motion이면 prev가 hist와 같은 모양이어야 한다'

        assert not self.depth_tag or H == self.history, (H, self.history)
        track = list(hist.unbind(1))
        feats = []
        for i in range(n_frames):
            idx = [max(0, i - k) for k in range(self.history - 1, -1, -1)]
            f = self.step(torch.stack([track[j] for j in idx], dim=1))
            parts = [f] + ([hist[:, min(i, H - 1)].to(f.dtype)] if self.anchor else [])
            if self.motion:
                a = min(i, H - 1)
                parts.append((hist[:, a] - prev[:, a]).to(f.dtype))
            if self.depth_tag:
                tag = f.new_zeros(f.size(0), N_TAGS)
                tag[:, tag_index(i, H)] = 1
                parts.append(tag)
            feats.append(torch.cat(parts, dim=-1))
            if i + 1 >= H and i + 1 >= len(track):
                track.append(f)
        return torch.stack(feats, dim=1)
