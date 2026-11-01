"""Interaction gate 구현 smoke test. 실제 data 없이 CPU에서 1~2분 안에 끝난다.

확인하는 것:
1. gate_mode=off이면 같은 weight의 FactorizedPredictor(CF-JEPA)와 출력이 같다.
2. gate_mode=learned도 학습 시작 시점(h_int 출력층 0 초기화)에는 CF-JEPA와 출력이 같다.
3. u -> c 경로: learned/always에서는 z^u를 바꾸면 z^c 예측이 바뀐다. off에서는 그대로다.
   action을 바꿔도 z^u 예측은 그대로다(h는 action을 보지 않는다).
4. α: learned는 (0, 1), always는 1, off는 0이고 shape이 (B, T, 1)이다.
5. zu_stopgrad=True이면 z^c 예측에서 z^u로 gradient가 가지 않는다.
6. Hydra로 model/cfjepa_gate.yaml을 만들고 scripts/train.py의 world_model_forward를 돌리면
   gate_sparsity_loss가 나오고 loss가 유한하며, gate와 h_int까지 gradient가 간다.
7. get_cost가 (B, S) cost를 돌려준다(CEM planning 경로).

사용: python scripts/colab/smoke_gate.py
"""

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT), str(ROOT / 'scripts')]

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import open_dict

from iawm.cfjepa_gate import FactorizedPredictor, InteractionGatedPredictor

B, T, D_C, D_U = 4, 3, 128, 64


def check(name, cond):
    print(('PASS ' if cond else 'FAIL ') + name)
    if not cond:
        raise SystemExit(1)


def make(mode, **kw):
    torch.manual_seed(0)
    p = InteractionGatedPredictor(num_frames=T, gate_mode=mode, **kw).eval()
    return p


def perturb_int_out(p):
    """0으로 초기화된 h_int 출력층을 흔들어 u -> c 경로가 실제로 열리게 한다."""
    with torch.no_grad():
        p.int_out.weight.normal_(0, 0.5)


def test_equivalence():
    torch.manual_seed(1)
    emb, act = torch.randn(B, T, D_C + D_U), torch.randn(B, T, D_C)
    base = FactorizedPredictor(num_frames=T).eval()
    for mode in ('off', 'learned'):
        p = make(mode)
        missing, unexpected = base.load_state_dict(p.state_dict(), strict=False)
        check(f'{mode}: CF-JEPA weight가 gated predictor 안에 모두 있다', not missing)
        with torch.no_grad():
            same = torch.allclose(p(emb, act), base(emb, act), atol=1e-6)
        check(f'{mode}: 학습 시작 시 출력 = CF-JEPA 출력', same)


def test_paths_and_alpha():
    torch.manual_seed(2)
    emb, act = torch.randn(B, T, D_C + D_U), torch.randn(B, T, D_C)
    emb2 = emb.clone()
    emb2[..., D_C:] += torch.randn(B, T, D_U)  # z^u만 바꾼다
    act2 = torch.randn(B, T, D_C)
    for mode, expect_u2c in (('learned', True), ('always', True), ('off', False)):
        p = make(mode)
        perturb_int_out(p)
        with torch.no_grad():
            out = p(emb, act)
            alpha = p.last_alpha.clone()
            out_u = p(emb2, act)
            out_a = p(emb, act2)
        changed = not torch.allclose(out[..., :D_C], out_u[..., :D_C], atol=1e-6)
        check(f'{mode}: z^u 변경이 z^c 예측을 바꾸는가 = {expect_u2c}', changed == expect_u2c)
        check(f'{mode}: action 변경은 z^u 예측을 바꾸지 않는다',
              torch.allclose(out[..., D_C:], out_a[..., D_C:], atol=1e-6))
        check(f'{mode}: alpha shape (B, T, 1)', tuple(alpha.shape) == (B, T, 1))
        if mode == 'learned':
            check('learned: 0 < alpha < 1', bool(((alpha > 0) & (alpha < 1)).all()))
        else:
            check(f'{mode}: alpha = {1.0 if mode == "always" else 0.0}',
                  bool((alpha == (1.0 if mode == 'always' else 0.0)).all()))


def test_stopgrad():
    for stop in (False, True):
        p = make('learned', zu_stopgrad=stop)
        perturb_int_out(p)
        emb = torch.randn(B, T, D_C + D_U, requires_grad=True)
        p(emb, torch.randn(B, T, D_C))[..., :D_C].sum().backward()
        g_u = emb.grad[..., D_C:].abs().sum().item()
        check(f'zu_stopgrad={stop}: z^c 예측 -> z^u gradient {"없음" if stop else "있음"}',
              (g_u == 0) if stop else (g_u > 0))


def build_model():
    with initialize_config_dir(config_dir=str(ROOT / 'configs/train'), version_base=None):
        cfg = compose(config_name='cfjepa', overrides=['model=cfjepa_gate', 'data=collision_tworoom'])
    with open_dict(cfg):
        cfg.model_kind = 'cfjepa'
        action_dim = cfg.data.dataset.frameskip * 2
        cfg.model.action_encoder.input_dim = action_dim
        cfg.model.action_dim = action_dim
    torch.manual_seed(0)
    return cfg, instantiate(cfg.model)


def test_train_forward(cfg, model):
    from stable_worldmodel.wm.loss import SIGReg
    from train import world_model_forward

    check('config: data 파일 이름이 collision_tworoom.h5', cfg.data.dataset.name == 'collision_tworoom.h5')
    check('config: predictor가 InteractionGatedPredictor',
          isinstance(model.predictor, InteractionGatedPredictor))
    check('config: model_kind 판정이 cfjepa', cfg.model._target_.endswith('CFJEPA'))
    perturb_int_out(model.predictor)  # h_int 쪽 gradient도 확인하려고 연다
    model.train()
    logged = {}
    fake = SimpleNamespace(
        model=model,
        sigreg=SIGReg(**cfg.loss.sigreg_kwargs),
        log_dict=lambda d, **_: logged.update(d),
    )
    n = cfg.data.dataset.num_steps
    batch = {
        'pixels': torch.randn(2, n, 3, cfg.img_size, cfg.img_size),
        'action': torch.randn(2, n, cfg.model.action_dim),
    }
    batch['action'][0, -1] = float('nan')  # 에피소드 경계
    out = world_model_forward(fake, batch, 'fit', cfg)
    for k in ('fwd_loss', 'inv_loss', 'adv_loss', 'sigreg_loss', 'gate_sparsity_loss', 'loss'):
        check(f'train: {k} 유한', k in out and torch.isfinite(out[k]).item())
    for k in ('fit/gate_alpha_mean', 'fit/gate_open_ratio', 'fit/gate_sparsity_loss'):
        check(f'train: {k} 로깅', k in logged)
    out['loss'].backward()
    pr = model.predictor
    check('train: gradient가 gate까지 간다', pr.gate[0].weight.grad is not None and pr.gate[0].weight.grad.abs().sum() > 0)
    check('train: gradient가 h_int까지 간다', pr.int_in.weight.grad is not None and pr.int_in.weight.grad.abs().sum() > 0)
    enc_grad = sum(p.grad.abs().sum() for p in model.encoder.parameters() if p.grad is not None)
    check('train: gradient가 encoder까지 간다', enc_grad > 0)


def test_get_cost(cfg, model):
    model.eval()
    S, H, horizon = 3, 1, 5
    a = cfg.model.action_dim
    info = {
        'pixels': torch.randn(1, S, H, 3, cfg.img_size, cfg.img_size),
        'goal': torch.randn(1, S, H, 3, cfg.img_size, cfg.img_size),
        'action': torch.randn(1, S, H, a),
    }
    with torch.no_grad():
        cost = model.get_cost(info, torch.randn(1, S, horizon, a))
    check('get_cost: (B, S) shape', tuple(cost.shape) == (1, S))
    check('get_cost: 유한', bool(torch.isfinite(cost).all()))


if __name__ == '__main__':
    test_equivalence()
    test_paths_and_alpha()
    test_stopgrad()
    cfg, model = build_model()
    test_train_forward(cfg, model)
    test_get_cost(cfg, model)
    print('ALL PASS')
