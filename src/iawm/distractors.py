"""CF-JEPA 방식의 visual distractor (CF-JEPA §IV-C-1).

논문 설정:
- 단일 색 사각형이 이미지의 무작위 가장자리 하나에 붙어서 나타난다.
- 가장자리에 수직인 길이는 49px, 가장자리 방향 폭은 [18, 36]px에서 고른다(224×224 기준).
- 가장자리를 따라 일정한 속도로 움직인다. 속도는 [3, 18] px/step에서 고른다.
- 기본 개수는 N = 2다.

논문에 없어서 우리가 정한 것 (docs/experiments.md E3에 기록):
- 색은 RGB 각 채널을 [0, 255]에서 균일하게 고른다.
- 이동 방향(+/-)은 반반으로 고른다.
- 사각형이 화면 끝에 닿으면 튕겨 돌아온다(reflect). 그래서 항상 N개가 모두 보인다.
- 나중에 고른 사각형이 앞에 그려진다.

같은 코드를 두 곳에서 쓴다.
- 학습: batch 전체에 GPU에서 덧그린다(`overlay`). 이미지가 ImageNet 정규화된 상태라
  색도 같은 정규화를 거쳐 칠한다.
- 평가: `envs/distractor_tworoom.py`가 env step마다 렌더링된 frame에 덧그린다.

시간 단위는 env step이다. TwoRoom dataset은 frameskip 5로 읽으므로, 학습 sample의
연속한 frame 사이에는 5 step이 흐른다. 학습 때는 frame index에 frameskip을 곱해서 넘긴다.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

EDGE_TOP, EDGE_BOTTOM, EDGE_LEFT, EDGE_RIGHT = 0, 1, 2, 3


@dataclass
class DistractorConfig:
    num: int = 2
    length: int = 49  # 가장자리에 수직인 길이 (px)
    width_range: tuple[int, int] = (18, 36)  # 가장자리 방향 폭 (px)
    speed_range: tuple[float, float] = (3.0, 18.0)  # px/step
    img_size: int = 224


def sample_params(
    batch: int,
    cfg: DistractorConfig,
    device=None,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """사각형 N개의 파라미터를 batch 단위로 고른다. 모든 값의 shape은 (B, N[, 3])이다."""
    shape = (batch, cfg.num)
    kw = dict(device=device, generator=generator)
    edge = torch.randint(0, 4, shape, **kw)
    lo, hi = cfg.width_range
    width = torch.randint(lo, hi + 1, shape, **kw).float()
    slo, shi = cfg.speed_range
    speed = slo + (shi - slo) * torch.rand(shape, **kw)
    sign = torch.randint(0, 2, shape, **kw).float() * 2 - 1
    start = torch.rand(shape, **kw) * (cfg.img_size - width)
    color = torch.randint(0, 256, (*shape, 3), **kw)
    return {
        'edge': edge,
        'width': width,
        'velocity': speed * sign,
        'start': start,
        'color': color,
    }


def along_edge_position(params: dict[str, torch.Tensor], t: torch.Tensor, img_size: int) -> torch.Tensor:
    """시각 t(step)에서 사각형의 가장자리 방향 시작 좌표. 끝에 닿으면 튕긴다.

    t: (T,) float. 반환: (B, T, N).
    """
    span = img_size - params['width']  # 움직일 수 있는 범위 [0, span]
    s = params['start'][:, None, :] + params['velocity'][:, None, :] * t[None, :, None]
    span = span[:, None, :].clamp(min=1e-6)
    m = torch.remainder(s, 2 * span)
    return span - (m - span).abs()  # 0..span 사이를 왕복하는 삼각파


def masks(params: dict[str, torch.Tensor], t: torch.Tensor, cfg: DistractorConfig) -> torch.Tensor:
    """사각형별 boolean mask. 반환: (B, T, N, H, W)."""
    size = cfg.img_size
    device = params['edge'].device
    pos = along_edge_position(params, t.to(device).float(), size)  # (B, T, N)
    coord = torch.arange(size, device=device, dtype=torch.float32)
    ys = coord.view(1, 1, 1, size, 1)
    xs = coord.view(1, 1, 1, 1, size)

    edge = params['edge'][:, None, :, None, None]  # (B, 1, N, 1, 1)
    horizontal = edge <= EDGE_BOTTOM  # 위·아래 가장자리: x 방향으로 움직인다
    along = torch.where(horizontal, xs, ys)
    perp = torch.where(horizontal, ys, xs)

    p = pos[..., None, None]
    w = params['width'][:, None, :, None, None]
    in_along = (along >= p) & (along < p + w)
    near_start = (edge == EDGE_TOP) | (edge == EDGE_LEFT)
    in_perp = torch.where(near_start, perp < cfg.length, perp >= size - cfg.length)
    return in_along & in_perp


def overlay(
    images: torch.Tensor,
    params: dict[str, torch.Tensor],
    t: torch.Tensor,
    cfg: DistractorConfig,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
) -> torch.Tensor:
    """images (B, T, C, H, W)에 사각형을 칠한 새 tensor를 돌려준다.

    mean/std가 주어지면 images가 (x/255 - mean)/std로 정규화되어 있다고 보고 색도 같은
    변환을 한다. 없으면 images가 0..255 값이라고 본다.
    """
    m = masks(params, t, cfg)  # (B, T, N, H, W)
    color = params['color'].to(images.dtype)  # (B, N, 3)
    if mean is not None:
        color = (color / 255.0 - mean.to(images)) / std.to(images)
    out = images.clone()
    for n in range(cfg.num):  # 나중 사각형이 앞에 온다
        c = color[:, n][:, None, :, None, None]  # (B, 1, C, 1, 1)
        out = torch.where(m[:, :, n][:, :, None], c, out)
    return out
