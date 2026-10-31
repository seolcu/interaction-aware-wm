"""CF-JEPA 구현 smoke test. 실제 data 없이 몇 초 안에 끝난다(CPU 가능).

확인하는 것:
1. Distractor: 사각형 면적, 화면 안에 머무는지, 정규화 공간에서 색이 맞는지.
2. Gradient reversal: backward에서 부호가 뒤집히는지.
3. Factorization: z^u를 바꿔도 z^c 예측이 그대로인지, action을 바꿔도 z^u 예측이 그대로인지.
4. Loss 4개가 모두 유한하고 backward가 되는지.
5. get_cost가 (B, S) cost를 돌려주고 z^c만 보는지.
6. Distractor env가 등록되고, N=0이면 원래 TwoRoom과 같은 frame을 내는지.

사용: python scripts/colab/smoke_cfjepa.py [--out DIR]   (DIR에 env frame 그림을 저장한다)
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import open_dict

from iawm.cfjepa import grad_reverse
from iawm.distractors import DistractorConfig, masks, overlay, sample_params


def check(name, cond):
    print(('PASS ' if cond else 'FAIL ') + name)
    if not cond:
        raise SystemExit(1)


def test_distractors():
    cfg = DistractorConfig(num=2)
    gen = torch.Generator().manual_seed(0)
    p = sample_params(64, cfg, generator=gen)
    t = torch.arange(0, 200, 5).float()
    m = masks(p, t, cfg)  # (B, T, N, H, W)
    area = m.sum(dim=(-1, -2)).float()
    expected = (p['width'] * cfg.length)[:, None, :].expand_as(area)
    check('distractor area = 49 x width (always fully on screen)', torch.allclose(area, expected))
    img = torch.zeros(64, len(t), 3, 224, 224)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3)
    out = overlay(img, p, t, cfg, mean, std)
    b, ti, n = 0, 3, 1
    ys, xs = torch.nonzero(m[b, ti, n], as_tuple=True)
    # 뒤에 그린 사각형(n=1) 픽셀의 색을 되돌려서 원래 RGB와 비교한다.
    rgb = (out[b, ti, :, ys[0], xs[0]] * std[0] + mean[0]) * 255
    check('normalized color round-trips to RGB', torch.allclose(rgb, p['color'][b, n].float(), atol=1e-3))


def test_grl():
    x = torch.randn(5, 3, requires_grad=True)
    grad_reverse(x, 1.0).sum().backward()
    check('gradient reversal flips sign', torch.allclose(x.grad, -torch.ones_like(x)))


def build_model():
    with initialize_config_dir(config_dir=str(ROOT / 'configs/train'), version_base=None):
        cfg = compose(config_name='cfjepa')
    with open_dict(cfg):
        cfg.model.action_dim = 10
        cfg.model.action_encoder.input_dim = 10
    return instantiate(cfg.model), cfg


def test_model():
    torch.manual_seed(0)
    model, cfg = build_model()
    model.eval()
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'params: {n:.2f}M')

    B, T = 4, 3
    emb = torch.randn(B, T, 192)
    act = torch.randn(B, T, 10)
    with torch.no_grad():
        a_emb = model.action_encoder(act)
        base = model.predict(emb, a_emb)
        emb2 = emb.clone()
        emb2[..., 128:] = torch.randn(B, T, 64)
        p_u_changed = model.predict(emb2, a_emb)
        p_a_changed = model.predict(emb, model.action_encoder(torch.randn(B, T, 10)))
    check('z^c prediction ignores z^u', torch.allclose(base[..., :128], p_u_changed[..., :128]))
    check('z^u prediction changes with z^u', not torch.allclose(base[..., 128:], p_u_changed[..., 128:]))
    check('z^u prediction ignores action', torch.allclose(base[..., 128:], p_a_changed[..., 128:]))

    # 학습 forward 한 번 (작은 이미지로 encoder까지 통과)
    model.train()
    from scripts.train import world_model_forward  # noqa: E402

    class Dummy:
        pass

    from stable_worldmodel.wm.loss import SIGReg

    self = Dummy()
    self.model = model
    self.sigreg = SIGReg(knots=17, num_proj=64)
    self.log_dict = lambda *a, **k: None
    with open_dict(cfg):
        cfg.model_kind = 'cfjepa'
    batch = {
        'pixels': torch.randn(2, 4, 3, 224, 224),
        'action': torch.randn(2, 4, 10),
    }
    batch['action'][1, -1] = float('nan')  # 에피소드 경계
    out = world_model_forward(self, batch, 'train', cfg)
    for k in ['fwd_loss', 'inv_loss', 'adv_loss', 'sigreg_loss', 'loss']:
        check(f'{k} finite ({out[k].item():.3f})', torch.isfinite(out[k]).item())
    out['loss'].backward()
    enc_grad = sum(p.grad.abs().sum().item() for p in model.encoder.parameters() if p.grad is not None)
    adv_grad = sum(p.grad.abs().sum().item() for p in model.adversary_head.parameters() if p.grad is not None)
    check('gradients reach encoder and adversary head', enc_grad > 0 and adv_grad > 0)

    # planning cost
    model.eval()
    S, H, horizon = 8, 1, 5
    info = {
        'pixels': torch.randn(1, S, H, 3, 224, 224),
        'goal': torch.randn(1, S, H, 3, 224, 224),
        'action': torch.zeros(1, S, H, 10),
    }
    cands = torch.randn(1, S, horizon, 10)
    with torch.no_grad():
        cost = model.get_cost(dict(info), cands)
        check(f'get_cost shape {tuple(cost.shape)} == (1, {S})', tuple(cost.shape) == (1, S))
        d = {'predicted_emb': torch.randn(1, S, 6, 192), 'goal_emb': torch.randn(1, 1, 192)}
        c1 = model.criterion(d)
        d['predicted_emb'][..., 128:] += 100.0
        check('cost uses z^c only', torch.allclose(c1, model.criterion(d)))


def test_env(out_dir):
    import gymnasium as gym

    import envs.distractor_tworoom  # noqa: F401
    from stable_worldmodel.envs.two_room.env import TwoRoomEnv

    env0 = gym.make('iawm/TwoRoomDistractor-v1', num_distractors=0).unwrapped
    ref = TwoRoomEnv()
    env0.reset(seed=3)
    ref.reset(seed=3)
    check('N=0 renders identical to TwoRoom', np.array_equal(env0.render(), ref.render()))

    env = gym.make('iawm/TwoRoomDistractor-v1', num_distractors=2).unwrapped
    env.reset(seed=1)
    frames = [env.render()]
    rng = np.random.default_rng(0)
    for _ in range(19):
        env.step(rng.uniform(-1, 1, 2).astype(np.float32))
        frames.append(env.render())
    check('distractor frames differ over time', not np.array_equal(frames[0], frames[5]))
    if out_dir:
        from PIL import Image

        out_dir.mkdir(parents=True, exist_ok=True)
        sheet = np.concatenate([np.concatenate(frames[i * 5:(i + 1) * 5], 1) for i in range(4)], 0)
        Image.fromarray(sheet).save(out_dir / 'distractor_env_sheet.png')
        print('saved', out_dir / 'distractor_env_sheet.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()
    test_distractors()
    test_grl()
    test_model()
    test_env(args.out)
    print('ALL PASS')
