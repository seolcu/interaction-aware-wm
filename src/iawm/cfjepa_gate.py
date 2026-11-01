"""CF-JEPA 재구현 (Byrd et al., arXiv 2610.00727).

공식 코드가 없어서 논문 §III-C, §III-D를 따라 LeWM(stable-worldmodel 0.1.1의
`stable_worldmodel.wm.lewm.LeWM`) 위에 다시 만든다. LeWM을 subclass하므로 `rollout`,
`get_cost`를 그대로 물려받고, 기존 CEM 평가 pipeline에 바로 들어간다.

논문과 같은 것:
- Encoder: ViT-Tiny(patch 14) + LeWM projector. z는 192-dim이고 z^c 128 + z^u 64로 나눈다.
- g: action을 AdaLN-zero로 받는 causal transformer. 입력 z^c, 출력 다음 z^c.
- h: action을 받지 않는 더 작은 causal transformer. 입력 z^u, 출력 다음 z^u.
- ψ (inverse): [z^c_t, z^c_{t+1}] -> a_t, 2-layer GELU MLP.
- φ (adversarial): gradient reversal + [z^u_t, z^u_{t+1}] -> a_t, 2-layer GELU MLP.
- Loss: L_fwd + inv_weight·L_inv + adv_weight·L_adv + sigreg_weight·L_SIGReg
  (논문 표기로 α, β, γ. 코드에서는 gate의 `alpha`와 헷갈리지 않게 이름을 바꿨다.)
- Planning cost: z^c만 쓴다. Eq. 7처럼 매 step의 ||z^c_k - z^c_goal||를 더한다.

논문에 없어서 우리가 정한 것:
- h의 크기: g(depth 6, heads 16)의 절반 정도(depth 3, heads 8).
- LeWM의 pred_proj(BatchNorm MLP)는 z^c와 z^u를 섞으므로 그대로 쓰지 않는다.
  z^c용, z^u용 pred_proj를 따로 둔다. 섞이면 "z^u는 z^c 예측에 쓰이지 않는다"는
  factorization이 깨진다.
- ψ, φ의 hidden 크기: 512.
- Action embedding은 g의 입력 차원(128)에 맞춘다(LeWM Transformer의 cond_proj가
  input_dim -> hidden_dim이기 때문).
"""

from __future__ import annotations

import torch
from einops import rearrange
from stable_worldmodel.wm.lewm import LeWM
from stable_worldmodel.wm.lewm.module import MLP, Block, Predictor, Transformer
from torch import nn


class GradientReversal(torch.autograd.Function):
    """forward는 그대로, backward는 gradient 부호를 뒤집는다 (Ganin & Lempitsky, 2015)."""

    @staticmethod
    def forward(ctx, x, scale: float):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.scale * grad, None


def grad_reverse(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return GradientReversal.apply(x, scale)


def two_layer_mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Module:
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))


class FactorizedPredictor(nn.Module):
    """[z^c, z^u] -> [g(z^c, a), h(z^u)]. 두 부분은 서로의 입력을 보지 않는다."""

    def __init__(
        self,
        *,
        num_frames: int = 3,
        dim_c: int = 128,
        dim_u: int = 64,
        # g: LeWM predictor와 같은 크기
        g_hidden_dim: int = 192,
        g_depth: int = 6,
        g_heads: int = 16,
        g_dim_head: int = 64,
        g_mlp_dim: int = 2048,
        # h: 더 작은 transformer (우리가 정한 크기)
        h_hidden_dim: int = 128,
        h_depth: int = 3,
        h_heads: int = 8,
        h_dim_head: int = 32,
        h_mlp_dim: int = 512,
        dropout: float = 0.1,
        proj_hidden_dim: int = 2048,
    ):
        super().__init__()
        self.num_frames = num_frames  # LeWM.rollout이 history 길이로 읽는다
        self.dim_c = dim_c
        self.dim_u = dim_u
        self.g = Predictor(
            num_frames=num_frames,
            input_dim=dim_c,
            hidden_dim=g_hidden_dim,
            output_dim=dim_c,
            depth=g_depth,
            heads=g_heads,
            dim_head=g_dim_head,
            mlp_dim=g_mlp_dim,
            dropout=dropout,
        )
        self.h_pos_embedding = nn.Parameter(torch.randn(1, num_frames, dim_u))
        self.h = Transformer(
            dim_u, h_hidden_dim, dim_u, h_depth, h_heads, h_dim_head, h_mlp_dim, dropout,
            block_class=Block,  # 조건 입력 없는 causal block
        )
        self.proj_c = MLP(dim_c, proj_hidden_dim, dim_c, norm_fn=nn.BatchNorm1d)
        self.proj_u = MLP(dim_u, proj_hidden_dim, dim_u, norm_fn=nn.BatchNorm1d)

    @staticmethod
    def _proj(proj: nn.Module, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        return rearrange(proj(rearrange(x, 'b t d -> (b t) d')), '(b t) d -> b t d', b=b)

    def forward(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        z_c, z_u = emb.split([self.dim_c, self.dim_u], dim=-1)
        T = emb.size(1)
        pred_c = self._proj(self.proj_c, self.g(z_c, act_emb))
        pred_u = self._proj(self.proj_u, self.h(z_u + self.h_pos_embedding[:, :T]))
        return torch.cat([pred_c, pred_u], dim=-1)


class CFJEPA(LeWM):
    def __init__(
        self,
        encoder,
        predictor: FactorizedPredictor,
        action_encoder,
        projector=None,
        action_dim: int = 10,
        head_hidden_dim: int = 512,
        grl_scale: float = 1.0,
        cost: str = 'sum_l2',
        **kwargs,
    ):
        # pred_proj는 FactorizedPredictor 안에 z^c, z^u별로 있으므로 여기서는 Identity다.
        super().__init__(encoder, predictor, action_encoder, projector=projector, pred_proj=None)
        assert cost in ('sum_l2', 'last_mse')
        self.dim_c = predictor.dim_c
        self.dim_u = predictor.dim_u
        self.grl_scale = grl_scale
        self.cost = cost
        self.inverse_head = two_layer_mlp(2 * self.dim_c, head_hidden_dim, action_dim)  # ψ
        self.adversary_head = two_layer_mlp(2 * self.dim_u, head_hidden_dim, action_dim)  # φ

    def split(self, emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return emb.split([self.dim_c, self.dim_u], dim=-1)

    def inverse(self, z_c_t: torch.Tensor, z_c_next: torch.Tensor) -> torch.Tensor:
        return self.inverse_head(torch.cat([z_c_t, z_c_next], dim=-1))

    def adversary(self, z_u_t: torch.Tensor, z_u_next: torch.Tensor) -> torch.Tensor:
        x = grad_reverse(torch.cat([z_u_t, z_u_next], dim=-1), self.grl_scale)
        return self.adversary_head(x)

    def criterion(self, info_dict: dict) -> torch.Tensor:
        """z^c만으로 cost를 계산한다. 반환: (B, S)."""
        pred_c = info_dict['predicted_emb'][..., : self.dim_c]  # (B, S, T, d_c)
        goal_emb = info_dict['goal_emb']
        if goal_emb.ndim == pred_c.ndim - 1:  # get_cost가 만든 goal_emb는 (B, T, D)다
            goal_emb = goal_emb[:, None]
        goal_c = goal_emb[..., -1:, : self.dim_c].detach()  # (B, 1|S, 1, d_c)
        if self.cost == 'sum_l2':  # 논문 Eq. 7
            return (pred_c - goal_c).norm(dim=-1).sum(dim=-1)
        # LeWM 방식: 마지막 step의 squared error
        return (pred_c[..., -1:, :] - goal_c).pow(2).sum(dim=(-2, -1))


# =====================================================================================
# 여기부터 interaction gate (강연우). 위쪽은 src/iawm/cfjepa.py를 그대로 복사한 것이다.
#
#   ẑ^c_{t+1} = g(z^c_t, a_t) + α_t · h_int(z^c_t, z^u_t, a_t)
#   α_t       = σ(MLP([z^c_t, z^u_t, Δz^u_t, a_t]))
#   L         = L_CF-JEPA + ρ · E[α_t]          (ρ는 scripts/train.py에서 더한다)
#
# README 수식과 다르게 정한 것:
# - h_int 출력을 int_scale · tanh(·)로 bound한다. bound가 없으면 h_int를 크게, α를 아주
#   작게 만들어 sparsity penalty를 공짜로 피할 수 있다. 그러면 α가 contact를 구분하지 못한다.
# - 게이트 입력에 Δz^u_t = z^u_t - z^u_{t-1}을 더한다. 한 frame의 z^u로는 obstacle이
#   다가오는지 멀어지는지 알 수 없다(첫 frame은 Δ = 0).
# - h_int의 마지막 층을 0으로 초기화한다. 학습 시작 시 출력이 CF-JEPA와 정확히 같다.
# =====================================================================================


class InteractionGatedPredictor(FactorizedPredictor):
    """FactorizedPredictor에 u -> c 경로를 gate α로 더한다.

    forward 시그니처와 반환값은 FactorizedPredictor와 같다. 그래서 CFJEPA, LeWM.rollout,
    CEM 평가 pipeline이 그대로 동작한다. α는 forward 뒤 `self.last_alpha`(B, T, 1)에 남는다.

    gate_mode:
    - 'learned': α = σ(MLP(...)). 제안 모델.
    - 'always':  α = 1. ablation (gate 없이 항상 반영).
    - 'off':     α = 0. h_int를 쓰지 않는다. 출력이 CF-JEPA와 같다.
    zu_stopgrad: h_int와 gate로 들어가는 z^u를 detach한다. ablation (z^u stop-gradient).
    """

    def __init__(
        self,
        *,
        gate_mode: str = 'learned',
        zu_stopgrad: bool = False,
        # h_int: 작은 action 조건 causal transformer (우리가 정한 크기)
        int_hidden_dim: int = 128,
        int_depth: int = 2,
        int_heads: int = 4,
        int_dim_head: int = 32,
        int_mlp_dim: int = 512,
        int_scale: float = 1.0,
        # gate MLP
        gate_hidden_dim: int = 256,
        gate_init_bias: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert gate_mode in ('learned', 'always', 'off')
        self.gate_mode = gate_mode
        self.zu_stopgrad = zu_stopgrad
        self.int_scale = int_scale
        d_c, d_u = self.dim_c, self.dim_u
        dropout = kwargs.get('dropout', 0.1)

        # LeWM Transformer의 cond_proj는 input_dim -> hidden_dim이라 action embedding 차원
        # (dim_c = 128)과 입력 차원이 같아야 한다. 그래서 [z^c, z^u]를 먼저 dim_c로 사영한다.
        self.int_in = nn.Linear(d_c + d_u, d_c)
        self.h_int = Predictor(
            num_frames=self.num_frames,
            input_dim=d_c,
            hidden_dim=int_hidden_dim,
            output_dim=d_c,
            depth=int_depth,
            heads=int_heads,
            dim_head=int_dim_head,
            mlp_dim=int_mlp_dim,
            dropout=dropout,
        )
        self.int_out = nn.Linear(d_c, d_c)
        nn.init.zeros_(self.int_out.weight)
        nn.init.zeros_(self.int_out.bias)

        self.gate = two_layer_mlp(d_c + 2 * d_u + d_c, gate_hidden_dim, 1)
        nn.init.constant_(self.gate[-1].bias, gate_init_bias)

        self.last_alpha: torch.Tensor | None = None

    def interaction(self, z_c, z_u, act_emb):
        """α · h_int. 반환: (B, T, d_c), α (B, T, 1)."""
        if self.zu_stopgrad:
            z_u = z_u.detach()
        if self.gate_mode == 'learned':
            dz_u = torch.cat([torch.zeros_like(z_u[:, :1]), z_u[:, 1:] - z_u[:, :-1]], dim=1)
            logit = self.gate(torch.cat([z_c, z_u, dz_u, act_emb], dim=-1))
            alpha = torch.sigmoid(logit.float()).to(z_c.dtype)
        else:
            alpha = z_c.new_ones(*z_c.shape[:2], 1)
        x = self.int_in(torch.cat([z_c, z_u], dim=-1))
        delta = self.int_scale * torch.tanh(self.int_out(self.h_int(x, act_emb)))
        return alpha * delta, alpha

    def forward(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        z_c, z_u = emb.split([self.dim_c, self.dim_u], dim=-1)
        T = emb.size(1)
        pred_c = self._proj(self.proj_c, self.g(z_c, act_emb))
        pred_u = self._proj(self.proj_u, self.h(z_u + self.h_pos_embedding[:, :T]))
        if self.gate_mode == 'off':
            self.last_alpha = z_c.new_zeros(*z_c.shape[:2], 1)
        else:
            corr, self.last_alpha = self.interaction(z_c, z_u, act_emb)
            pred_c = pred_c + corr
        return torch.cat([pred_c, pred_u], dim=-1)
