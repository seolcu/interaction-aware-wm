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


class Forecaster(nn.Module):
    def __init__(self, dim: int = 192, history: int = 3, hidden: int = 1024, depth: int = 3):
        super().__init__()
        self.dim, self.history = dim, history
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
    def plan_features(self, hist: torch.Tensor, n_frames: int) -> torch.Tensor:
        """planning용 feature. hist (B, H, dim) 실제 history latent. 반환 (B, n_frames, dim).

        feature[i] = frame i까지(history 밖이면 forecaster가 굴린 frame) 보고 예측한 frame i+1.
        history보다 앞의 frame이 필요하면 첫 frame을 반복한다(학습 column과 같다).
        """
        H = hist.size(1)
        track = list(hist.unbind(1))
        feats = []
        for i in range(n_frames):
            idx = [max(0, i - k) for k in range(self.history - 1, -1, -1)]
            f = self.step(torch.stack([track[j] for j in idx], dim=1))
            feats.append(f)
            if i + 1 >= H and i + 1 >= len(track):
                track.append(f)
        return torch.stack(feats, dim=1)
