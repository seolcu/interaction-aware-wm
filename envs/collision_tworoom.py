"""Collision env: 움직이는 obstacle이 agent를 미는 TwoRoom.

stable-worldmodel의 TwoRoomEnv를 subclass한다. 바꾸는 것은 obstacle 하나뿐이다.

- Obstacle은 action과 agent의 영향을 받지 않고 혼자 움직인다(uncontrollable).
  영향은 obstacle -> agent 한 방향뿐이라, CF-JEPA의 independence assumption
  ("uncontrollable part never affects the controllable part")만 깨진다.
- Obstacle은 테두리와 중앙 벽에서 반사되어 자기 방에 머문다. 문은 지나가지 않는다.
- num_obstacles=0이면 부모 TwoRoomEnv와 궤적이 완전히 같다.
- Goal image(_target_img)에는 obstacle을 그리지 않는다. 부모 reset() 안에서
  그려지는데, 그 시점에는 obstacle이 아직 없다.
- import하면 gym ID `iawm/CollisionTwoRoom-v1`로 등록된다. Variation space에는 등록하지 않았다.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from stable_worldmodel.envs.two_room.env import TwoRoomEnv


class CollisionTwoRoomEnv(TwoRoomEnv):
    OBSTACLE_COLOR = np.array([0, 90, 255], dtype=np.uint8)

    def __init__(
        self,
        *args,
        num_obstacles: int = 1,
        obstacle_radius: float = 14.0,
        obstacle_speed: tuple[float, float] = (3.0, 6.0),
        spawn_mode: str = 'crossing',
        obstacle_info: str = 'all',
        **kwargs,
    ):
        assert spawn_mode in ('crossing', 'random')
        self.num_obstacles = int(num_obstacles)
        self.obstacle_radius = float(obstacle_radius)
        self.obstacle_speed = tuple(float(s) for s in obstacle_speed)
        self.spawn_mode = spawn_mode
        # 'first': info에 첫 obstacle만 넣는다. 평가 pipeline은 info buffer를 dataset(obstacle 1개)
        # 모양으로 잡아서, obstacle이 여러 개면 shape가 맞지 않는다(E14).
        assert obstacle_info in ('all', 'first')
        self.obstacle_info = obstacle_info
        self.obstacle_pos = torch.zeros((0, 2), dtype=torch.float32)
        self.obstacle_vel = torch.zeros((0, 2), dtype=torch.float32)
        self.contact = False
        self.pinned = False
        super().__init__(*args, **kwargs)

    # ---------------- Gym API ----------------

    def reset(self, seed=None, options=None):
        self.obstacle_pos = torch.zeros((0, 2), dtype=torch.float32)
        self.obstacle_vel = torch.zeros((0, 2), dtype=torch.float32)
        self.contact = False
        self.pinned = False
        obs, info = super().reset(seed=seed, options=options)
        self._spawn_obstacles()
        return obs, self._get_info() | {'distance_to_target': info['distance_to_target']}

    def step(self, action):
        action_t = torch.clamp(torch.as_tensor(action, dtype=torch.float32), -1.0, 1.0)
        speed = float(self.variation_space['agent']['speed'].value.item())
        prev = self.agent_position
        pos = prev + action_t * speed

        self._move_obstacles()
        # 벽 clamp가 agent를 다시 obstacle 안으로 넣을 수 있어 push와 clamp를 몇 번 반복한다.
        # 그래도 겹치면 obstacle과 벽 사이에 끼인 것(pinned)으로 본다.
        self.contact = False
        for _ in range(3):
            pos, pushed = self._push_agent(pos)
            pos = self._apply_collisions(prev, pos)
            self.contact |= pushed
            if not self._overlap(pos).any():
                break
        self.agent_position = pos
        self.pinned = bool(self._overlap(pos).any())

        dist = float(torch.norm(self.agent_position - self.target_position))
        terminated = dist < 16.0  # 부모와 같은 기준
        info = self._get_info()
        info['distance_to_target'] = dist
        return self._get_obs(), 0.0, terminated, False, info

    # ---------------- 평가 pipeline (World.evaluate) ----------------

    def _set_goal_state(self, goal_state):
        """평가 때 World가 _set_state, _set_goal_state 순서로 시작 위치와 목표를 dataset 값으로 바꾼다.

        reset 때 놓은 obstacle은 바뀌기 전 위치 기준이므로, 목표까지 정해진 뒤 다시 놓는다.
        Goal image에 obstacle이 그려지지 않도록 먼저 비우고 부모를 부른다.
        """
        self.obstacle_pos = torch.zeros((0, 2), dtype=torch.float32)
        self.obstacle_vel = torch.zeros((0, 2), dtype=torch.float32)
        super()._set_goal_state(goal_state)
        self.contact = False
        self.pinned = False
        self._spawn_obstacles()

    # ---------------- Obstacles ----------------

    def _room_bounds(self, pos: torch.Tensor, r: float):
        """pos가 속한 방의 [lo, hi] (x, y). 반지름 r만큼 안쪽으로 줄인다."""
        bs = float(self.BORDER_SIZE)
        lo = [bs + r, bs + r]
        hi = [self.IMG_SIZE - bs - r, self.IMG_SIZE - bs - r]
        ax = 0 if self.wall_axis == 1 else 1  # vertical wall -> x축으로 나뉜다
        half = self.wall_thickness // 2
        c = float(self.WALL_CENTER)
        if float(pos[ax]) < c:
            hi[ax] = c - half - r
        else:
            lo[ax] = c + half + r
        return torch.tensor(lo), torch.tensor(hi)

    def _sample_speed(self):
        lo, hi = self.obstacle_speed
        return float(self.np_random.uniform(lo, hi))

    def _spawn_obstacles(self):
        n = self.num_obstacles
        if n == 0:
            return
        r = self.obstacle_radius
        a = self.agent_position
        lo, hi = self._room_bounds(a, r)
        pos, vel = [], []
        for _ in range(n):
            if self.spawn_mode == 'crossing':
                # agent -> target 경로 위 한 점을 고르고, 경로에 수직으로 떨어진 곳에서
                # 그 점을 향해 움직이게 한다. 같은 방 안으로 clamp한다.
                d = self.target_position - a
                d_norm = float(torch.norm(d)) + 1e-6
                u = d / d_norm
                perp = torch.tensor([-u[1], u[0]])
                t = float(self.np_random.uniform(25.0, 60.0))
                anchor = a + u * min(t, d_norm)
                side = 1.0 if self.np_random.random() < 0.5 else -1.0
                offset = float(self.np_random.uniform(35.0, 60.0))
                p = anchor + side * perp * offset
                p = torch.maximum(torch.minimum(p, hi), lo)
                v = anchor - p
                v = v / (float(torch.norm(v)) + 1e-6) * self._sample_speed()
            else:
                p = lo + torch.as_tensor(self.np_random.random(2), dtype=torch.float32) * (hi - lo)
                ang = float(self.np_random.uniform(0, 2 * np.pi))
                v = torch.tensor([np.cos(ang), np.sin(ang)], dtype=torch.float32) * self._sample_speed()
            pos.append(p.to(torch.float32))
            vel.append(v.to(torch.float32))
        self.obstacle_pos = torch.stack(pos)
        self.obstacle_vel = torch.stack(vel)

    def _move_obstacles(self):
        r = self.obstacle_radius
        for i in range(self.obstacle_pos.shape[0]):
            lo, hi = self._room_bounds(self.obstacle_pos[i], r)
            p = self.obstacle_pos[i] + self.obstacle_vel[i]
            for k in range(2):
                if p[k] < lo[k]:
                    p[k] = 2 * lo[k] - p[k]
                    self.obstacle_vel[i, k] = -self.obstacle_vel[i, k]
                elif p[k] > hi[k]:
                    p[k] = 2 * hi[k] - p[k]
                    self.obstacle_vel[i, k] = -self.obstacle_vel[i, k]
            self.obstacle_pos[i] = p

    def _agent_radius(self):
        return float(self.variation_space['agent']['radius'].value.item())

    def _overlap(self, pos: torch.Tensor, tol: float = 1e-3):
        if self.obstacle_pos.shape[0] == 0:
            return torch.zeros(0, dtype=torch.bool)
        d = torch.norm(self.obstacle_pos - pos, dim=1)
        return d < self._agent_radius() + self.obstacle_radius - tol

    def _push_agent(self, pos: torch.Tensor):
        """겹친 obstacle마다 agent를 중심선 방향으로 밀어낸다."""
        contact = False
        min_d = self._agent_radius() + self.obstacle_radius
        for i in range(self.obstacle_pos.shape[0]):
            d = pos - self.obstacle_pos[i]
            dist = float(torch.norm(d))
            if dist < min_d:
                contact = True
                n = d / dist if dist > 1e-6 else self.obstacle_vel[i] / (float(torch.norm(self.obstacle_vel[i])) + 1e-6)
                pos = self.obstacle_pos[i] + n * (min_d + 0.5)
        return pos, contact

    def render_past(self, steps_back: int) -> np.ndarray:
        """episode 시작 `steps_back` step 전의 화면 (E26, planning history용).

        "agent가 시작 위치에 멈춘 채 obstacle을 지켜보았다"고 보고, obstacle만 속도를 뒤집어 거꾸로
        움직인 위치에 그린다. 반사도 거꾸로 하면 원래 궤적과 같다. 상태(위치, 속도)는 바꾸지 않는다.
        """
        if self.obstacle_pos.shape[0] == 0 or steps_back <= 0:
            return self.render()
        pos0, vel0 = self.obstacle_pos.clone(), self.obstacle_vel.clone()
        try:
            self.obstacle_vel = -self.obstacle_vel
            for _ in range(steps_back):
                self._move_obstacles()
            return self.render()
        finally:
            self.obstacle_pos, self.obstacle_vel = pos0, vel0

    # ---------------- Info / Rendering ----------------

    def _get_info(self):
        info = super()._get_info()
        k = 1 if self.obstacle_info == 'first' else None
        info['obstacle_pos'] = self.obstacle_pos[:k].detach().cpu().numpy().copy()
        info['obstacle_vel'] = self.obstacle_vel[:k].detach().cpu().numpy().copy()
        if k == 1 and self.obstacle_pos.shape[0] == 0:
            # num_obstacles=0 대조군 평가(E24): 평가 buffer 모양(obstacle 1개)에 맞춘다.
            info['obstacle_pos'] = np.zeros((1, 2), dtype=np.float32)
            info['obstacle_vel'] = np.zeros((1, 2), dtype=np.float32)
        info['contact'] = bool(self.contact)
        info['pinned'] = bool(self.pinned)
        return info

    def _render_frame(self, agent_pos: torch.Tensor):
        img = super()._render_frame(agent_pos)
        for p in self.obstacle_pos:
            dist = torch.sqrt((self.grid_x - float(p[0])) ** 2 + (self.grid_y - float(p[1])) ** 2)
            alpha = (self.obstacle_radius + 0.5 - dist).clamp(0, 1)  # 가장자리 1px anti-alias
            img = self._alpha_blend(img, alpha, self.OBSTACLE_COLOR)
        return img


ENV_ID = 'iawm/CollisionTwoRoom-v1'
if ENV_ID not in gym.registry:
    gym.register(id=ENV_ID, entry_point=f'{__name__}:CollisionTwoRoomEnv')
