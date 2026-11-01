"""Distractor env: CF-JEPA식 visual distractor가 있는 PushT.

stable-worldmodel의 PushT(`swm/PushT-v1`)에 `DistractorMixin`을 붙인다. TwoRoom 버전과 같은
사각형(길이 49px, 폭 18~36px, 속도 3~18px/step, 기본 N=2)을 `render()` frame에만 그린다.

- PushT는 reset 안에서 `self._goal = self.render()`로 goal image를 만든다. Mixin이 그 시점에
  사각형 parameter를 비워 두므로 goal image는 깨끗하다.
- 논문은 "task를 가리지 않게" 배치했다고만 적었다. 우리는 TwoRoom과 같이 가장자리에 붙인다.
  T block이 가장자리 근처에 있으면 일부가 가려질 수 있다.

import하면 `iawm/PushTDistractor-v1`로 gym에 등록된다.
"""

from __future__ import annotations

import gymnasium as gym
from stable_worldmodel.envs.pusht.env import PushT

from envs.distractor_mixin import DistractorMixin

ENV_ID = 'iawm/PushTDistractor-v1'


class DistractorPushTEnv(DistractorMixin, PushT):
    pass


if ENV_ID not in gym.registry:
    gym.register(id=ENV_ID, entry_point=f'{__name__}:DistractorPushTEnv')
