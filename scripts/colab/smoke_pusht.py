"""Distractor PushT 확인 (E7).

1. N=0이면 원래 swm/PushT-v1과 같은 frame이 나온다.
2. N=2이면 frame에 사각형이 있고, goal image(info['goal'] 또는 env._goal)에는 없다.
3. 연속 20 step frame sheet를 /content/pusht_distractor_sheet.png로 저장한다.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import stable_worldmodel  # noqa: E402,F401
from PIL import Image  # noqa: E402

import envs.distractor_pusht  # noqa: E402,F401

base = gym.make('swm/PushT-v1').unwrapped
d0 = gym.make('iawm/PushTDistractor-v1', num_distractors=0).unwrapped
d2 = gym.make('iawm/PushTDistractor-v1', num_distractors=2).unwrapped

rng = np.random.default_rng(0)
acts = rng.uniform(-1, 1, size=(20, 2)).astype(np.float32)
for e in (base, d0, d2):
    e.reset(seed=7)
f_base, f0, f2 = [], [], []
for a in acts:
    for e, buf in ((base, f_base), (d0, f0), (d2, f2)):
        e.step(a)
        buf.append(e.render())
same0 = all(np.array_equal(a, b) for a, b in zip(f_base, f0))
diff2 = np.mean([np.mean(np.any(a != b, axis=-1)) for a, b in zip(f_base, f2)])
goal_clean = np.array_equal(base._goal, d2._goal)
print(f'N=0 identical to PushT: {same0}')
print(f'N=2 changed pixel fraction (mean over 20 steps): {diff2:.3f}')
print(f'N=2 goal image identical to PushT goal: {goal_clean}')
assert same0 and goal_clean and diff2 > 0.01

rows = [np.concatenate(f2[i:i + 5], axis=1) for i in range(0, 20, 5)]
Image.fromarray(np.concatenate(rows, axis=0)).save('/content/pusht_distractor_sheet.png')
print('SMOKE OK')
