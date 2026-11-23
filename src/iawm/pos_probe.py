"""z^c에서 agent와 obstacle 위치(px)를 읽는 probe (E39 px 진단, E40 planning 진단).

E38에서 Collision data로 학습한 CF-JEPA의 z^c에는 agent 위치뿐 아니라 obstacle 위치도 들어 있었다.
이 probe는 두 가지에 쓴다.
- 진단: 모델이 예측한 z^c를 px 위치로 읽어, 예측이 agent 위치를 얼마나 맞히는지 본다.
- planning cost (`cost=agent_px`, 진단용): z^c 거리 대신 "z^c에서 읽은 agent 위치와 goal image에서 읽은 agent
  위치의 거리"를 cost로 쓴다. z^c에 섞인 obstacle 성분이 cost에서 빠진다. probe를 위치 label로 학습하므로
  제안 방법이 아니라 "cost가 병목인가"를 가르는 도구다.

출력 순서: [agent x, agent y, obstacle x, obstacle y].
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class PosProbe(nn.Module):
    def __init__(self, in_dim: int = 128, hidden: int = 256, out_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, out_dim))
        for name, dim in (('xm', in_dim), ('xs', in_dim), ('ym', out_dim), ('ys', out_dim)):
            self.register_buffer(name, torch.ones(dim) if name.endswith('s') else torch.zeros(dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net((z.float() - self.xm) / self.xs) * self.ys + self.ym


def fit(X, Y, device, steps: int = 6000, hidden: int = 256, seed: int = 0):
    """X (N, d_c) latent, Y (N, 4) px (numpy). 마지막 5%는 학습에 쓰지 않는다.

    반환: 학습된 PosProbe(eval mode), (held-out agent 오차 px, obstacle 오차 px).
    """
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(X), generator=g)
    n_te = max(1000, len(X) // 20)
    te, tr = idx[:n_te].to(device), idx[n_te:].to(device)
    X, Y = torch.from_numpy(X).float().to(device), torch.from_numpy(Y).float().to(device)
    torch.manual_seed(seed)
    probe = PosProbe(X.shape[1], hidden, Y.shape[1]).to(device)
    probe.xm.copy_(X[tr].mean(0)), probe.xs.copy_(X[tr].std(0) + 1e-6)
    probe.ym.copy_(Y[tr].mean(0)), probe.ys.copy_(Y[tr].std(0) + 1e-6)
    opt = torch.optim.AdamW(probe.net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        b = tr[torch.randint(0, len(tr), (2048,), device=device)]
        loss = (probe.net((X[b] - probe.xm) / probe.xs) - (Y[b] - probe.ym) / probe.ys).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    probe.eval().requires_grad_(False)
    with torch.no_grad():
        d = probe(X[te]) - Y[te]
    return probe, (d[:, :2].norm(dim=-1).mean().item(), d[:, 2:].norm(dim=-1).mean().item())


def save(probe: PosProbe, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = {'in_dim': probe.xm.numel(), 'hidden': probe.net[0].out_features, 'out_dim': probe.ym.numel()}
    torch.save({'config': cfg, 'state_dict': probe.state_dict()}, path)


def load(path: Path, device='cpu') -> PosProbe:
    ck = torch.load(path, map_location='cpu', weights_only=True)
    probe = PosProbe(**ck['config'])
    probe.load_state_dict(ck['state_dict'])
    return probe.to(device).eval().requires_grad_(False)


def agent_px_cost(model, info_dict: dict) -> torch.Tensor:
    """planning cost: 예측한 z^c에서 읽은 agent 위치와 goal에서 읽은 agent 위치의 거리(px)를 모든 step에 대해 더한다.

    CFJEPA.criterion의 sum_l2와 같은 꼴이고 거리만 z^c 공간에서 px 공간으로 바꾼 것이다. 반환: (B, S).
    """
    probe = getattr(model, 'pos_probe', None)
    assert probe is not None, "cost=agent_px는 model.pos_probe가 필요하다 (scripts/eval.py의 +pos_probe=<파일>)"
    pred_c = info_dict['predicted_emb'][..., : model.dim_c]  # (B, S, T, d_c)
    goal_emb = info_dict['goal_emb']
    if goal_emb.ndim == pred_c.ndim - 1:
        goal_emb = goal_emb[:, None]
    goal_c = goal_emb[..., -1:, : model.dim_c].detach()
    p = probe(pred_c)[..., :2]
    g = probe(goal_c)[..., :2]
    return (p - g).norm(dim=-1).sum(dim=-1).to(pred_c.dtype)
