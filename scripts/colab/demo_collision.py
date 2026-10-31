"""Collision env prototype 확인: sanity check, contact 통계, demo GIF.

사용 (repo root를 PYTHONPATH에 둔다):
  PYTHONPATH=/content/iawm python demo_collision.py --out /content/outputs/collision_tworoom
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from stable_worldmodel.envs.two_room.env import TwoRoomEnv

from envs.collision_tworoom import CollisionTwoRoomEnv


def scripted_action(env, rng, noise=0.3):
    """Goal로 직진하되, 다른 방이면 첫 번째 문을 먼저 지난다. TwoRoom data 수집 방식과 같은 계열."""
    a, g = env.agent_position.numpy(), env.target_position.numpy()
    ax = 0 if env.wall_axis == 1 else 1
    c = float(env.WALL_CENTER)
    waypoint = g
    if (a[ax] < c) != (g[ax] < c):
        door = np.zeros(2, dtype=np.float32)
        door[ax] = c
        door[1 - ax] = float(env.door_positions[0])
        if abs(a[1 - ax] - door[1 - ax]) > 4.0 or abs(a[ax] - c) > 25.0:
            waypoint = door
    d = waypoint - a
    d = d / (np.linalg.norm(d) + 1e-6)
    return np.clip(d + rng.normal(0, noise, 2), -1, 1).astype(np.float32)


def check_parent_equivalence(n_steps=200, seed=0):
    """num_obstacles=0이면 부모 TwoRoomEnv와 궤적이 같아야 한다."""
    p, c = TwoRoomEnv(), CollisionTwoRoomEnv(num_obstacles=0)
    p.reset(seed=seed)
    c.reset(seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(n_steps):
        act = rng.uniform(-1, 1, 2).astype(np.float32)
        p.step(act)
        c.step(act)
        assert torch.equal(p.agent_position, c.agent_position), (p.agent_position, c.agent_position)
    print(f'[check] num_obstacles=0 궤적이 부모와 같다 ({n_steps} steps)')


def in_wall(env, pos):
    wall, _ = env._wall_and_door_masks()
    x, y = int(round(float(pos[0]))), int(round(float(pos[1])))
    return bool(wall[y, x])


def run_episode(env, seed, max_steps, rng, frames=None):
    env.reset(seed=seed)
    contacts, pinned = [], 0
    reached = False
    for t in range(max_steps):
        _, _, term, _, info = env.step(scripted_action(env, rng))
        contacts.append(info['contact'])
        pinned += info['pinned']
        # sanity 1: 밀려난 뒤 겹침은 벽에 끼인(pinned) 경우만 허용
        assert info['pinned'] or not env._overlap(env.agent_position).any(), f'overlap at seed={seed} t={t}'
        # sanity 2: push가 agent를 벽 안으로 넣지 않는다. contact step만 본다.
        # 부모 TwoRoom도 문을 지날 때 agent 중심이 벽 가장자리에 걸치는 일이 있어서다
        # (scripted policy 200 episode 중 10개, obstacle 없이 확인).
        assert not (info['contact'] and in_wall(env, env.agent_position)), f'push into wall at seed={seed} t={t}'
        if frames is not None:
            # 시각화 전용 표시다. env가 내보내는 observation에는 goal을 그리지 않는다(TwoRoom 기본값).
            img = Image.fromarray(env.render())
            draw = ImageDraw.Draw(img)
            gx, gy = (float(v) for v in env.target_position)
            draw.line([gx - 7, gy - 7, gx + 7, gy + 7], fill=(0, 170, 0), width=3)
            draw.line([gx - 7, gy + 7, gx + 7, gy - 7], fill=(0, 170, 0), width=3)
            if info['contact']:
                draw.rectangle([0, 0, 223, 223], outline=(255, 160, 0), width=6)
            frames.append(img)
        if term:
            reached = True
            break
    return np.array(contacts), pinned, reached


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='outputs/collision_tworoom')
    ap.add_argument('--episodes', type=int, default=200)
    ap.add_argument('--max-steps', type=int, default=100)
    ap.add_argument('--gifs', type=int, default=4)
    ap.add_argument('--spawn-mode', default='crossing')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    check_parent_equivalence()

    env = CollisionTwoRoomEnv(spawn_mode=args.spawn_mode)
    ep_contact, step_contact, total_steps, pinned_steps, reached = 0, 0, 0, 0, 0
    gif_seeds = []
    for ep in range(args.episodes):
        c, p, r = run_episode(env, ep, args.max_steps, np.random.default_rng(ep))
        ep_contact += bool(c.any())
        step_contact += int(c.sum())
        total_steps += len(c)
        pinned_steps += p
        reached += r
        if c.any() and len(gif_seeds) < args.gifs:
            gif_seeds.append(ep)
    print(f'[check] overlap / wall assert 통과 ({args.episodes} episodes)')

    stats = {
        'spawn_mode': args.spawn_mode,
        'episodes': args.episodes,
        'episode_contact_rate': ep_contact / args.episodes,
        'step_contact_rate': step_contact / total_steps,
        'pinned_step_rate': pinned_steps / total_steps,
        'goal_reached_rate': reached / args.episodes,
        'mean_episode_len': total_steps / args.episodes,
    }
    print(json.dumps(stats, indent=2))
    (out / 'stats.json').write_text(json.dumps(stats, indent=2))

    for s in gif_seeds:
        frames = []
        run_episode(env, s, args.max_steps, np.random.default_rng(s), frames)  # 통계 때와 같은 episode
        frames[0].save(out / f'demo_seed{s}.gif', save_all=True, append_images=frames[1:], duration=100, loop=0)
    print('[gif]', [f'demo_seed{s}.gif' for s in gif_seeds])


if __name__ == '__main__':
    main()
