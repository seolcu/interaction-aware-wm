"""Distractor env 공통 부분: 어떤 gym env든 `render()` frame에만 CF-JEPA식 사각형을 덧그린다.

사용: `class DistractorFooEnv(DistractorMixin, FooEnv)`, 생성자에 `num_distractors`를 넘긴다.
- 사각형은 그림일 뿐이라 상태와 dynamics는 부모 env와 완전히 같다. env step마다 움직인다.
- 사각형 parameter는 부모 reset()이 끝난 뒤 env의 np_random으로 뽑는다. 부모 reset() 안에서
  그리는 goal image에는 사각형이 들어가지 않는다(reset 시작에서 parameter를 비운다).
- 모양과 움직임은 학습 augmentation(`iawm.distractors`)과 같은 함수를 쓴다.
"""

from __future__ import annotations

import torch

from iawm.distractors import DistractorConfig, overlay, sample_params


class DistractorMixin:
    def __init__(self, *args, num_distractors: int = 2, **kwargs):
        self.distractor_cfg = DistractorConfig(num=int(num_distractors))
        self._distractor_params = None
        self._distractor_t = 0
        super().__init__(*args, **kwargs)

    def reset(self, seed=None, options=None):
        self._distractor_params = None  # 부모 reset 안의 goal image에 이전 사각형이 그려지지 않게
        obs, info = super().reset(seed=seed, options=options)
        gen = torch.Generator().manual_seed(int(self.np_random.integers(2**31 - 1)))
        self._distractor_params = sample_params(1, self.distractor_cfg, generator=gen)
        self._distractor_t = 0
        return obs, info

    def step(self, action):
        out = super().step(action)
        self._distractor_t += 1
        return out

    def render(self):
        img = super().render()  # (H, W, 3) uint8
        if self._distractor_params is None or self.distractor_cfg.num == 0:
            return img
        x = torch.from_numpy(img).permute(2, 0, 1)[None, None].float()  # (1, 1, 3, H, W)
        t = torch.tensor([float(self._distractor_t)])
        x = overlay(x, self._distractor_params, t, self.distractor_cfg)
        return x[0, 0].permute(1, 2, 0).to(torch.uint8).numpy()
