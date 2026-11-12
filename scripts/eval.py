"""Planning 평가 script. LeWM의 eval.py(github.com/lucas-maes/le-wm, commit 8edfeb3, MIT)를 고쳐 쓴다.

바꾼 것:
- repo의 src/와 envs/를 import path에 넣고, Distractor TwoRoom과 Collision TwoRoom env를 등록한다.
- 결과 파일을 $STABLEWM_HOME/results/<output.filename>에 쓴다.
- `+cost=last_mse`로 CF-JEPA의 planning cost를 평가 때 바꿀 수 있다(학습과 무관).
- `+gate_mode=off`로 gate 모델의 h_int를 끄고 평가할 수 있다(같은 encoder로 gate 효과만 비교).
- `+obs_fix=true`: 첫 관측을 dataset frame 대신 실제 env 화면으로 쓴다(E26, Collision env에서 필요).
- `+plan_history=3`: planning에 과거 frame 3장(frameskip 간격)과 그 사이 action을 넣는다(E26, Collision env 전용).

사용 예:
    python scripts/eval.py policy=cfjepa_tworoom_n2/weights_epoch_10.pt world.num_distractors=2
    python scripts/eval.py policy=lewm-tworooms world.num_distractors=0   # E0 재현
    python scripts/eval.py --config-name=collision_tworoom policy=<checkpoint>   # Collision TwoRoom
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
import envs.collision_tworoom  # noqa: E402,F401  (gym 등록)
import envs.distractor_pusht  # noqa: E402,F401  (gym 등록)
import envs.distractor_tworoom  # noqa: E402,F401  (gym 등록)

os.environ["MUJOCO_GL"] = "egl"

import time

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm

def img_transform(cfg):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def make_history_policy(base_cls, history: int, frameskip: int, obs_fix: bool):
    """WorldModelPolicy에 두 가지를 더한다 (E26).

    obs_fix: World._evaluate_from_dataset은 env를 dataset 시작 상태로 맞춘 뒤 첫 관측 pixels를 dataset
        frame으로 덮어쓴다. 원래 TwoRoom에서는 env 화면과 같지만, Collision env는 obstacle을 새로 놓으므로
        첫 계획이 실제와 다른 obstacle 배치를 보고 세워진다. episode 첫 step에서 pixels를 env.render()로 바꾼다.
    history > 1: 계획할 때 frame t-(history-1)·fs, ..., t-fs, t와 그 사이에 실행한 action block을 넣는다
        (학습과 같은 frameskip 간격). episode 시작 전 frame은 env.render_past로 만들고 action은 0이다
        ("시작 위치에 멈춰서 지켜보았다"). action은 process['action']으로 학습과 같이 정규화한다.
    """

    class HistoryPolicy(base_cls):
        def set_env(self, env):
            super().set_env(env)
            n = getattr(env, 'num_envs', 1)
            self.frames = [[] for _ in range(n)]   # env step마다의 frame (HWC uint8), 시작 전 frame 포함
            self.raw_actions = [[] for _ in range(n)]  # env step마다 실행한 raw action, 시작 전은 0
            self.started = np.zeros(n, dtype=bool)

        def _unwrapped(self, i):
            return self.env.envs[i].unwrapped

        def get_action(self, info_dict, **kwargs):
            info_dict = dict(info_dict)
            pix = np.array(info_dict['pixels'], copy=True)  # (n, 1, H, W, C)
            n = pix.shape[0]
            pre = (history - 1) * frameskip
            for i in range(n):
                if not self.started[i]:
                    env = self._unwrapped(i)
                    if obs_fix:
                        pix[i, -1] = np.asarray(env.render())
                    if pre > 0:
                        for k in range(pre, 0, -1):
                            self.frames[i].append(np.asarray(env.render_past(k)))
                            self.raw_actions[i].append(np.zeros(self.env.single_action_space.shape, np.float32))
                    self.started[i] = True
                self.frames[i].append(pix[i, -1].copy())
            info_dict['pixels'] = pix
            if history > 1:
                idx = [-1 - frameskip * j for j in range(history - 1, -1, -1)]  # t-(h-1)fs ... t
                info_dict['pixels'] = np.stack(
                    [np.stack([self.frames[i][j] for j in idx]) for i in range(n)]
                )
                acts = []
                for i in range(n):
                    a = np.stack(self.raw_actions[i][-pre:])  # (pre, A): frame t-(h-1)fs부터 t 직전까지
                    if 'action' in self.process:
                        a = self.process['action'].transform(a)
                    acts.append(a.reshape(history - 1, -1))  # (h-1, fs*A), 학습 data와 같은 순서
                info_dict['hist_action'] = np.stack(acts).astype(np.float32)
            if not getattr(self, '_logged', False):  # 첫 호출에서 입력 모양을 한 번 남긴다 (smoke test 확인용)
                self._logged = True
                print('[history policy] pixels', info_dict['pixels'].shape,
                      'hist_action', getattr(info_dict.get('hist_action'), 'shape', None),
                      'frames/env', len(self.frames[0]), 'obs_fix', obs_fix, flush=True)
            action = super().get_action(info_dict, **kwargs)
            for i in range(n):
                a = np.asarray(action[i], dtype=np.float32)
                self.raw_actions[i].append(np.nan_to_num(a))  # 끝난 env는 nan -> 0 (더 쓰이지 않는다)
            return action

    return HistoryPolicy


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"

    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )
    return dataset

@hydra.main(version_base=None, config_path="../configs/eval", config_name="tworoom")
def run(cfg: DictConfig):
    """Run evaluation of dinowm vs random policy."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    # create world environment
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    # create the transform
    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset  # get_dataset(cfg, cfg.dataset.stats)
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(stats_dataset.get_col_data(col_name), return_index=True)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != "action":
            process[f"goal_{col}"] = process[col]

    # -- run evaluation
    policy = cfg.get("policy", "random")

    if policy != "random":
        model = swm.wm.utils.load_pretrained(cfg.policy)
        model = model.to("cuda")
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        if cfg.get("cost") is not None:  # CF-JEPA planning cost 바꾸기: sum_l2(논문 Eq. 7) | last_mse
            model.cost = cfg.cost
        if cfg.get("gate_mode") is not None:  # gate 모델의 α 바꾸기: learned | always | off (E12, off는 CF-JEPA 부분만)
            model.predictor.gate_mode = cfg.gate_mode
        config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        plan_history = int(cfg.get("plan_history", 1))
        obs_fix = bool(cfg.get("obs_fix", False))
        policy_cls = swm.policy.WorldModelPolicy
        if plan_history > 1 or obs_fix:  # E26: 첫 관측을 실제 env 화면으로, planning에 과거 frame 넣기
            policy_cls = make_history_policy(
                policy_cls, plan_history, cfg.plan_config.action_block, obs_fix
            )
        policy = policy_cls(
            solver=solver, config=config, process=process, transform=transform
        )

    else:
        policy = swm.policy.RandomPolicy()

    results_path = Path(swm.data.utils.get_cache_dir(), "results", Path(cfg.output.filename).stem)

    # sample the episodes and the starting indices
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    # Map each dataset row’s episode_idx to its max_start_idx
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )

    # remove all the lines of dataset for which dataset['step_idx'] > max_start_per_row
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), "valid starting points found for evaluation.")

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
    )

    # sort increasingly to avoid issues with HDF5Dataset indexing
    random_episode_indices = np.sort(valid_indices[random_episode_indices])

    print(random_episode_indices)

    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    world.set_policy(policy)

    results_path.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video=results_path,
    )
    end_time = time.time()
    
    print(metrics)

    results_path = results_path / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("a") as f:
        f.write("\n")  # separate from previous runs

        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")

        f.write("==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")


if __name__ == "__main__":
    run()
