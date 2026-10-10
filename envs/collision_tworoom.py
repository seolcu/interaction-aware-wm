"""Env. ②: 움직이는 obstacle이 agent를 미는 TwoRoom.

stable-worldmodel의 TwoRoomEnv를 subclass한다. 바꾸는 것은 obstacle 하나뿐이다.

- Obstacle은 action과 agent의 영향을 받지 않고 혼자 움직인다(uncontrollable).
  영향은 obstacle -> agent 한 방향뿐이라, CF-JEPA의 independence assumption
  ("uncontrollable part never affects the controllable part")만 깨진다.
- Obstacle은 테두리와 중앙 벽에서 반사되어 자기 방에 머문다. 문은 지나가지 않는다.
- num_obstacles=0이면 부모 TwoRoomEnv와 궤적이 완전히 같다.
- Goal image(_target_img)에는 obstacle을 그리지 않는다. 부모 reset() 안에서
  그려지는데, 그 시점에는 obstacle이 아직 없다.
- Variation space와 gym ID에는 등록하지 않았다.
"""

from __future__ import annotations

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
        **kwargs,
    ):
        assert spawn_mode in ('crossing', 'random')
        self.num_obstacles = int(num_obstacles)
        self.obstacle_radius = float(obstacle_radius)
        self.obstacle_speed = tuple(float(s) for s in obstacle_speed)
        self.spawn_mode = spawn_mode
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

    # ---------------- Info / Rendering ----------------

    def _get_info(self):
        info = super()._get_info()
        info['obstacle_pos'] = self.obstacle_pos.detach().cpu().numpy().copy()
        info['obstacle_vel'] = self.obstacle_vel.detach().cpu().numpy().copy()
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
