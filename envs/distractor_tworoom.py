"""Distractor env: CF-JEPA식 visual distractor가 있는 TwoRoom.

stable-worldmodel의 TwoRoomEnv를 subclass하고, `render()`가 돌려주는 frame에만
사각형을 덧그린다. 사각형은 그림일 뿐이라 agent와 충돌하지 않고, 상태와 dynamics는
부모와 완전히 같다. 사각형은 env step마다 움직인다.

- 모양과 움직임은 `iawm.distractors`를 학습 augmentation과 함께 쓴다.
- Goal image(`_target_img`)에는 그리지 않는다. 평가 pipeline은 dataset frame을 goal로
  쓰므로 goal은 원래 깨끗하다(E3 결정 사항).
- 평가 시작 frame은 World가 dataset frame으로 덮어쓴다. 그래서 첫 observation에는
  사각형이 없고, 다음 step부터 보인다.

import하면 `iawm/TwoRoomDistractor-v1`로 gym에 등록된다.
"""

from __future__ import annotations

import gymnasium as gym
import torch
from stable_worldmodel.envs.two_room.env import TwoRoomEnv

from iawm.distractors import DistractorConfig, overlay, sample_params

ENV_ID = 'iawm/TwoRoomDistractor-v1'


class DistractorTwoRoomEnv(TwoRoomEnv):
    def __init__(self, *args, num_distractors: int = 2, **kwargs):
        self.distractor_cfg = DistractorConfig(num=int(num_distractors))
        self._distractor_params = None
        self._distractor_t = 0
        super().__init__(*args, **kwargs)

    def reset(self, seed=None, options=None):
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


if ENV_ID not in gym.registry:
    gym.register(id=ENV_ID, entry_point=f'{__name__}:DistractorTwoRoomEnv')
