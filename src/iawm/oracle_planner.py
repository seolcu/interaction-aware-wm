"""정답 simulator로 계획하는 CEM planner (E28, Collision env 진단 전용).

학습된 world model 대신 env의 실제 물리(agent 이동, 벽 clamp, obstacle push)를 batch로 다시 계산해서 계획한다.
같은 CEM 설정(후보 300, 반복 30, top-k 30, 25 step 계획, 25 step 실행)에서 성공률의 상한을 잰다.

- mode='aware': obstacle 궤적과 push를 simulator에 넣는다(완벽한 interaction 모델).
- mode='blind': obstacle을 빼고 계획한다. "obstacle은 agent에 영향을 주지 않는다"는 CF-JEPA 가정과 같은 모델이다.

두 성공률의 차이가 interaction을 정확히 예측했을 때 planning이 얻을 수 있는 최대 이득이다.
Obstacle은 agent와 action의 영향을 받지 않으므로, 궤적을 한 번 계산해 모든 후보가 같이 쓴다.
Cost는 CF-JEPA cost(블록 끝 latent와 goal의 거리 합)를 따라 블록 끝 agent 위치와 target의 거리 합이다.
"""

from __future__ import annotations

import numpy as np
import torch
from stable_worldmodel.policy import BasePolicy


class OraclePlanner(BasePolicy):
    def __init__(self, mode='aware', horizon=25, receding=25, action_block=5,
                 samples=300, iters=30, topk=30, seed=0, **kwargs):
        super().__init__(**kwargs)
        assert mode in ('aware', 'blind')
        self.type = f'oracle_{mode}'
        self.mode = mode
        self.horizon, self.receding, self.block = horizon, receding, action_block
        self.samples, self.iters, self.topk = samples, iters, topk
        self.gen = torch.Generator().manual_seed(seed)

    def set_env(self, env):
        self.env = env
        n = getattr(env, 'num_envs', 1)
        self.queue = [[] for _ in range(n)]
        self.expect = [None] * n  # 직전 action 실행 뒤 simulator가 예측한 위치
        self.sim_err = []  # 예측 위치와 실제 위치의 차이 (aware면 0에 가까워야 한다)

    def set_seed(self, seed):
        self.gen = torch.Generator().manual_seed(int(seed))

    # ---------- batched 물리 (envs/collision_tworoom.py, TwoRoomEnv._apply_collisions와 같은 계산) ----------

    @staticmethod
    def obstacle_trajectory(e, steps):
        """앞으로 steps step 동안 obstacle 위치 (steps, M, 2). env 상태는 그대로 둔다."""
        pos0, vel0 = e.obstacle_pos.clone(), e.obstacle_vel.clone()
        out = []
        try:
            for _ in range(steps):
                e._move_obstacles()
                out.append(e.obstacle_pos.clone())
        finally:
            e.obstacle_pos, e.obstacle_vel = pos0, vel0
        if not out or pos0.shape[0] == 0:
            return torch.zeros((steps, 0, 2)), torch.zeros((0, 2))
        return torch.stack(out), vel0

    @staticmethod
    def clamp(e, prev, p):
        bs = float(e.BORDER_SIZE)
        r = float(e.variation_space['agent']['radius'].value.item())
        p = p.clamp(bs + r, e.IMG_SIZE - bs - r)
        half = e.wall_thickness // 2
        c = float(e.WALL_CENTER)
        ax = 0 if e.wall_axis == 1 else 1  # 벽에 수직인 축
        ox = 1 - ax  # 문 위치를 재는 축
        door = torch.zeros(p.shape[0], dtype=torch.bool)
        for i in range(e.num_doors):
            dc, ds = float(e.door_positions[i]), float(e.door_sizes[i])
            door |= (p[:, ox] >= dc - ds - 1.75) & (p[:, ox] <= dc + ds + 1.75)
        lo_side = prev[:, ax] < c
        eff_lo, eff_hi = c - half - r, c + half + r
        new = p[:, ax].clone()
        new = torch.where(lo_side & (p[:, ax] > eff_lo) & ~door, torch.full_like(new, eff_lo - 0.5), new)
        new = torch.where(~lo_side & (p[:, ax] < eff_hi) & ~door, torch.full_like(new, eff_hi + 0.5), new)
        p = p.clone()
        p[:, ax] = new
        return p

    @staticmethod
    def push(e, p, obs, vel):
        min_d = float(e.variation_space['agent']['radius'].value.item()) + e.obstacle_radius
        for i in range(obs.shape[0]):
            d = p - obs[i]
            dist = d.norm(dim=1, keepdim=True)
            vdir = vel[i] / (vel[i].norm() + 1e-6)
            n = torch.where(dist > 1e-6, d / dist.clamp_min(1e-12), vdir.expand_as(d))
            p = torch.where(dist < min_d, obs[i] + n * (min_d + 0.5), p)
        return p

    def rollout_cost(self, e, actions, obs_traj, vel_traj, traj=None):
        """actions (S, T, 2) -> 블록 끝 target 거리의 합 (S,). traj가 list면 step마다 위치를 담는다."""
        speed = float(e.variation_space['agent']['speed'].value.item())
        target = e.target_position.float()
        S, T, _ = actions.shape
        p = e.agent_position.float().expand(S, 2).clone()
        cost = torch.zeros(S)
        for t in range(T):
            prev = p
            p = prev + actions[:, t] * speed
            if self.mode == 'aware' and obs_traj.shape[1] > 0:
                for _ in range(3):  # env.step과 같이 push와 clamp를 반복한다 (겹침이 없으면 둘 다 그대로다)
                    p = self.push(e, p, obs_traj[t], vel_traj[t])
                    p = self.clamp(e, prev, p)
            else:
                p = self.clamp(e, prev, p)
            if traj is not None:
                traj.append(p[0].clone())
            if (t + 1) % self.block == 0:
                cost += (p - target).norm(dim=1)
        return cost

    def plan(self, e):
        T = self.horizon
        obs_traj, _ = self.obstacle_trajectory(e, T)
        # push 방향 fallback용 속도: 반사로 바뀌므로 step마다 위치 차이로 근사한다
        if obs_traj.shape[1] > 0:
            prev = torch.cat([e.obstacle_pos[None], obs_traj[:-1]])
            vel_traj = obs_traj - prev
        else:
            vel_traj = obs_traj
        mean = torch.zeros(T, 2)
        std = torch.ones(T, 2)
        for _ in range(self.iters):
            a = (mean + std * torch.randn(self.samples, T, 2, generator=self.gen)).clamp(-1, 1)
            a[0] = mean.clamp(-1, 1)
            cost = self.rollout_cost(e, a, obs_traj, vel_traj)
            elite = a[cost.topk(self.topk, largest=False).indices]
            mean, std = elite.mean(0), elite.std(0) + 1e-3
        plan = mean.clamp(-1, 1)
        traj = []
        self.rollout_cost(e, plan[None], obs_traj, vel_traj, traj)
        return plan.numpy(), torch.stack(traj).numpy()

    def get_action(self, info_dict, **kwargs):
        n = len(self.queue)
        out = np.zeros((n, 2), dtype=np.float32)
        for i in range(n):
            e = self.env.envs[i].unwrapped
            if self.expect[i] is not None:
                self.sim_err.append(float(np.linalg.norm(e.agent_position.numpy() - self.expect[i])))
            if not self.queue[i]:
                plan, traj = self.plan(e)
                self.queue[i] = list(zip(plan[: self.receding], traj[: self.receding]))
            out[i], self.expect[i] = self.queue[i].pop(0)
        return out
