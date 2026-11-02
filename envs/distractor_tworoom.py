"""Distractor env: CF-JEPA식 visual distractor가 있는 TwoRoom.

stable-worldmodel의 TwoRoomEnv에 `DistractorMixin`을 붙인다. `render()` frame에만 사각형을
덧그리고, 충돌하지 않으며 dynamics는 부모와 같다.

- Goal image에는 그리지 않는다. 평가 pipeline은 dataset frame을 goal로 쓰므로 goal은 원래
  깨끗하다(E3 결정 사항).
- 평가 시작 frame은 World가 dataset frame으로 덮어쓴다. 그래서 첫 observation에는
  사각형이 없고, 다음 step부터 보인다.

import하면 `iawm/TwoRoomDistractor-v1`로 gym에 등록된다.
"""

from __future__ import annotations

import gymnasium as gym
from stable_worldmodel.envs.two_room.env import TwoRoomEnv

from envs.distractor_mixin import DistractorMixin

ENV_ID = 'iawm/TwoRoomDistractor-v1'


class DistractorTwoRoomEnv(DistractorMixin, TwoRoomEnv):
    pass


if ENV_ID not in gym.registry:
    gym.register(id=ENV_ID, entry_point=f'{__name__}:DistractorTwoRoomEnv')
