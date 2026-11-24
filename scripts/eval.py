"""Planning 평가 script. LeWM의 eval.py(github.com/lucas-maes/le-wm, commit 8edfeb3, MIT)를 고쳐 쓴다.

바꾼 것:
- repo의 src/와 envs/를 import path에 넣고, Distractor TwoRoom과 Collision TwoRoom env를 등록한다.
- 결과 파일을 $STABLEWM_HOME/results/<output.filename>에 쓴다.
- `+cost=last_mse`로 CF-JEPA의 planning cost를 평가 때 바꿀 수 있다(학습과 무관).
- `+gate_mode=off`로 gate 모델의 h_int를 끄고 평가할 수 있다(같은 encoder로 gate 효과만 비교).
- `+obs_fix=true`: 첫 관측을 dataset frame 대신 실제 env 화면으로 쓴다(E26, Collision env에서 필요).
- `+plan_history=3`: planning에 과거 frame 3장(frameskip 간격)과 그 사이 action을 넣는다(E26, Collision env 전용).
- `policy=oracle_aware|oracle_blind|oracle_avoid`: 학습된 모델 대신 정답 simulator로 CEM 계획한다(E28, Collision env 진단 전용).
- `+replan=contact` (oracle) / `+replan=gate +replan_tau=τ` (gate 모델) /
  `+replan=random +replan_p=p` (대조군: block마다 확률 p로 끊는다): 충돌이 예상되는 block까지만 실행하고 다시 계획한다(E37).
- `+contact_cost=λ`: gate 모델의 상상 α로 충돌 회피 cost를 더한다(E29). oracle_avoid에서는 정답 충돌에 같은 cost를 준다.
- `+cost=agent_px +pos_probe=<파일>`: planning cost를 z^c 거리 대신 z^c에서 읽은 agent 위치의 거리(px)로 바꾼다(E40 진단,
  iawm.pos_probe). probe는 scripts/fit_pos_probe.py가 만든다.
- `+plan_log=<npz 경로>`: 계획마다 고른 action 열의 예측 z^c와, env step마다 실제 agent 위치를 기록한다(E40).
  scripts/analyze_plan_log.py가 읽어 "planner가 고른 계획에서 예측이 맞았는가"를 낸다.
- `+action_clip=true`: CEM 후보 action을 모델에 넣기 전에 env의 action 범위로 자른다(E41). env는 `step()`에서 raw
  action을 [-1, 1]로 자르는데, CEM은 정규화한 action 공간에서 범위 없이 sample한다. 자르지 않으면 모델은 env가 실행할 수
  없는 action으로 예측한다. 정규화가 affine이므로 정규화 공간에서 자르면 env의 clamp와 정확히 같다. 실행하는 action(CEM의
  mean)은 그대로 두고 env가 자른다. plan log에는 고른 action과 범위 밖 비율을 함께 남긴다(설정과 무관하게).
- 정답 obstacle 입력 모델(predictor.int_oracle_key=obs_oracle, E39): planning 때 env에서 obstacle의 과거·현재·미래 상태를
  계산해 h_int에 넣는다. `+plan_history`가 모델의 history 길이와 같아야 한다. 설정 없이 모델을 보고 켠다.

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


def make_history_policy(base_cls, history: int, frameskip: int, obs_fix: bool, oracle_horizon: int = 0,
                        plan_log: bool = False, clip_history: bool = False, motion: bool = False,
                        oracle_noise: float = 0.0, noise_seed: int = 0, est=None, est_horizon: int = 0):
    """WorldModelPolicy에 두 가지를 더한다 (E26).

    obs_fix: World._evaluate_from_dataset은 env를 dataset 시작 상태로 맞춘 뒤 첫 관측 pixels를 dataset
        frame으로 덮어쓴다. 원래 TwoRoom에서는 env 화면과 같지만, Collision env는 obstacle을 새로 놓으므로
        첫 계획이 실제와 다른 obstacle 배치를 보고 세워진다. episode 첫 step에서 pixels를 env.render()로 바꾼다.
    history > 1: 계획할 때 frame t-(history-1)·fs, ..., t-fs, t와 그 사이에 실행한 action block을 넣는다
        (학습과 같은 frameskip 간격). episode 시작 전 frame은 env.render_past로 만들고 action은 0이다
        ("시작 위치에 멈춰서 지켜보았다"). action은 process['action']으로 학습과 같이 정규화한다.
    oracle_horizon > 0 (E39): 정답 obstacle 상태를 info의 'obs_oracle' (n, history + horizon - 1, 6K)로 넣는다.
        history frame의 상태는 frame과 같이 기록해 둔 값이고, 미래는 env에서 obstacle만 앞으로 돌려 계산한다
        (obstacle은 action과 무관하게 움직이므로 후보 action과 상관없이 정확하다). iawm.cfjepa_gate.CFJEPA.rollout이 읽는다.
    plan_log (E40): env step마다 [agent 위치, target 위치, contact]를 기록하고, 이번 호출에서 새로 계획할 env와
        그 시점의 step 번호를 남긴다(wrap_solver_count_and_gate가 계획의 예측과 묶는다).
    oracle_noise (E52): 정답 obstacle 입력의 속도에 frame마다 N(0, σ²) px/step 오차를 더한다(iawm.obs_oracle.vel_noise).
        계획할 때마다 새로 뽑는다. 학습 때(train_int_frozen.py --oracle-noise)와 같은 σ를 준다.
    est (E56): (탐지기, 예측기, device). history frame마다 (1 step 전 frame, frame)으로 obstacle을 탐지해
        info['obs_est'] (n, history + est_horizon - 1, 7·K_MAX)를 만든다(iawm.obs_detector.plan_features). 첫 계획에도
        1 step 전 frame이 있도록 시작 전 frame을 하나 더 render_past로 만든다. 지금 frame의 탐지를 env의 정답과 비교해
        self.est_err에 모은다(진단 출력).
    motion (E50): history frame마다 1 env step 전 frame을 info['pixels_prev']로 넣는다. 첫 계획에서도 있도록 시작 전 frame을
        하나 더 render_past로 만든다(그 action도 0). iawm.cfjepa_gate.CFJEPA.rollout이 encode해 forecaster feature를 만든다.
    """
    from iawm import obs_oracle as oo

    def _np(t):
        return np.asarray(t.detach().cpu().numpy() if torch.is_tensor(t) else t, dtype=np.float32).reshape(-1)

    class HistoryPolicy(base_cls):
        def set_env(self, env):
            super().set_env(env)
            n = getattr(env, 'num_envs', 1)
            self.frames = [[] for _ in range(n)]   # env step마다의 frame (HWC uint8), 시작 전 frame 포함
            self.raw_actions = [[] for _ in range(n)]  # env step마다 실행한 raw action, 시작 전은 0
            self.obs_states = [[] for _ in range(n)]   # frames와 같은 index의 obstacle (pos, vel) (E39)
            self.pos_log = [[] for _ in range(n)]      # env step마다 [agent xy, target xy, contact] (E40)
            self._plan_envs, self._plan_t = [], []
            self.started = np.zeros(n, dtype=bool)
            self._noise_rng = np.random.default_rng(noise_seed)  # E52
            self.est_err = {'pos': [], 'vel': [], 'miss': 0, 'fp': 0, 'n': 0}  # E56

        def _unwrapped(self, i):
            return self.env.envs[i].unwrapped

        def get_action(self, info_dict, **kwargs):
            info_dict = dict(info_dict)
            pix = np.array(info_dict['pixels'], copy=True)  # (n, 1, H, W, C)
            n = pix.shape[0]
            pre = (history - 1) * frameskip
            extra = 1 if (motion or est is not None) else 0
            track_obs = oracle_horizon > 0 or est is not None
            for i in range(n):
                if not self.started[i]:
                    env = self._unwrapped(i)
                    if obs_fix:
                        pix[i, -1] = np.asarray(env.render())
                    if pre > 0:
                        for k in range(pre + extra, 0, -1):
                            self.frames[i].append(np.asarray(env.render_past(k)))
                            if track_obs:
                                self.obs_states[i].append(oo.past_state(env, k))
                            self.raw_actions[i].append(np.zeros(self.env.single_action_space.shape, np.float32))
                    self.started[i] = True
                self.frames[i].append(pix[i, -1].copy())
                if track_obs:
                    self.obs_states[i].append(oo.current_state(self._unwrapped(i)))
            if plan_log:
                for i in range(n):
                    e = self._unwrapped(i)
                    self.pos_log[i].append(np.concatenate(
                        [_np(e.agent_position), _np(e.target_position), [float(getattr(e, 'contact', False))]]))
                # WorldModelPolicy.get_action과 같은 규칙으로, 이번 호출에서 새로 계획할 env를 미리 구한다
                flush = info_dict.get('_needs_flush')
                flush = np.zeros(n, bool) if flush is None else np.asarray(flush, bool)
                term = info_dict.get('terminated')
                dead = np.zeros(n, bool) if term is None else np.asarray(term, bool)
                self._plan_envs = [i for i in range(n) if (len(self._action_buffer[i]) == 0 or flush[i]) and not dead[i]]
                self._plan_t = [len(self.pos_log[i]) - 1 for i in self._plan_envs]
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
                if motion:
                    info_dict['pixels_prev'] = np.stack(
                        [np.stack([self.frames[i][j - 1] for j in idx]) for i in range(n)]
                    )
                if oracle_horizon > 0:
                    info_dict['obs_oracle'] = np.stack([
                        oo.plan_oracle([self.obs_states[i][j] for j in idx], self._unwrapped(i), oracle_horizon, frameskip)
                        for i in range(n)
                    ])
                # E57: 탐지는 이번 호출에서 계획을 세울 env가 있을 때만 한다(계획하지 않는 step에서는 obs_est를 쓰지 않는다).
                # planning 6개를 동시에 돌리면 매 step의 작은 GPU 호출이 밀려 실행이 3배 느려졌다. 계획에 들어가는 값은 같다.
                if est is not None:
                    _fl = info_dict.get('_needs_flush')
                    _fl = np.zeros(n, bool) if _fl is None else np.asarray(_fl, bool).reshape(n, -1)[:, -1]
                    _tm = info_dict.get('terminated')
                    _dd = np.zeros(n, bool) if _tm is None else np.asarray(_tm, bool).reshape(n, -1)[:, -1]
                    est_needed = any((len(self._action_buffer[i]) == 0 or _fl[i]) and not _dd[i] for i in range(n))
                if est is not None and est_needed:  # E56: 화면에서 obstacle 상태를 읽는다
                    from iawm import obs_detector as od
                    det, fc, dev = est
                    cur = np.stack([self.frames[i][j] for i in range(n) for j in idx])
                    prv = np.stack([self.frames[i][j - 1] for i in range(n) for j in idx])
                    P, V, M = det.detect(torch.from_numpy(prv).to(dev), torch.from_numpy(cur).to(dev))
                    P, V, M = (t.cpu().numpy().reshape(n, len(idx), *t.shape[1:]) for t in (P, V, M))
                    feats = []
                    term = info_dict.get('terminated')
                    dead = np.zeros(n, bool) if term is None else np.asarray(term, bool).reshape(n, -1)[:, -1]
                    for i in range(n):
                        feats.append(od.plan_features([(P[i, h], V[i, h], M[i, h]) for h in range(len(idx))], fc, est_horizon, dev))
                        if dead[i] or np.array_equal(self.frames[i][-1], self.frames[i][-2]):  # 끝난 env는 화면이 멈춰 있다(진단에서 뺀다)
                            continue
                        tp, tv = self.obs_states[i][-1]
                        pe, ve, miss, fp = od.match_error(P[i, -1], V[i, -1], M[i, -1], tp.reshape(-1, 2), tv.reshape(-1, 2))
                        self.est_err['pos'] += pe; self.est_err['vel'] += ve
                        self.est_err['miss'] += miss; self.est_err['fp'] += fp; self.est_err['n'] += len(tp) // 2
                    info_dict['obs_est'] = np.stack(feats).astype(np.float32)
                if oracle_horizon > 0:
                    if oracle_noise > 0:  # E52
                        k = info_dict['obs_oracle'].shape[-1] // 6
                        info_dict['obs_oracle'] = oo.vel_noise(info_dict['obs_oracle'], oracle_noise, k, self._noise_rng, frameskip)
            if not getattr(self, '_logged', False):  # 첫 호출에서 입력 모양을 한 번 남긴다 (smoke test 확인용)
                self._logged = True
                print('[history policy] pixels', info_dict['pixels'].shape,
                      'hist_action', getattr(info_dict.get('hist_action'), 'shape', None),
                      'frames/env', len(self.frames[0]), 'obs_fix', obs_fix,
                      'obs_oracle', getattr(info_dict.get('obs_oracle'), 'shape', None),
                      'pixels_prev', getattr(info_dict.get('pixels_prev'), 'shape', None),
                      'oracle_noise', oracle_noise, 'obs_est', getattr(info_dict.get('obs_est'), 'shape', None), flush=True)
            action = super().get_action(info_dict, **kwargs)
            for i in range(n):
                a = np.asarray(action[i], dtype=np.float32)
                a = np.nan_to_num(a)  # 끝난 env는 nan -> 0 (더 쓰이지 않는다)
                if clip_history:  # E41: 과거 action도 env가 실제로 실행한 값(잘린 값)으로 넣는다
                    a = np.clip(a, -1.0, 1.0)
                self.raw_actions[i].append(a)
            return action

    return HistoryPolicy


def action_bounds(process, action_block, low=-1.0, high=1.0):
    """env의 raw action 범위 [low, high]를 정규화 공간으로 옮긴다. 반환: (lo, hi) (action_block * A,) tensor.

    CEM의 action 차원은 (block, A)를 펼친 것이다(policy가 (H, block, A)로 되돌린다)라 A 차원 값을 block번 이어 붙인다.
    """
    sc = process['action']
    a, b = sc.transform(np.array([[low, low]], np.float32)), sc.transform(np.array([[high, high]], np.float32))
    lo, hi = np.minimum(a, b)[0], np.maximum(a, b)[0]
    return torch.tensor(np.tile(lo, action_block), dtype=torch.float32), torch.tensor(np.tile(hi, action_block), dtype=torch.float32)


def clip_model_actions(model, lo, hi):
    """E41: model.get_cost가 받는 후보 action을 [lo, hi]로 자른다 (env의 clamp를 모델 입력에서 재현)."""
    orig = model.get_cost

    def get_cost(info_dict, action_candidates):
        l, h = lo.to(action_candidates.device, action_candidates.dtype), hi.to(action_candidates.device, action_candidates.dtype)
        return orig(info_dict, torch.maximum(torch.minimum(action_candidates, h), l))

    model.get_cost = get_cost


def wrap_solver_count_and_gate(policy, model, tau, rand_p=None, seed=0, plan_log=False, bounds=None):
    """E37: 계획 횟수를 세고, tau가 있으면 gate α로 실행 구간을 자른다(gate-triggered replanning).

    CEM이 고른 계획(최종 mean)을 한 번 더 상상 rollout해서 block마다 gate α를 읽는다. history 3 frame +
    후보 5 block이면 predictor를 5번 부르고, k번째 호출의 α는 block k를 실행하는 동안의 contact 확률이다.
    α가 처음 tau를 넘는 block까지만 실행하고 그 다음 step에서 실제 관측으로 다시 계획한다(없으면 전부 실행).
    """
    solver = policy.solver
    orig_solve = solver.solve
    rng = np.random.default_rng(seed + 12345)
    policy.n_plans = 0
    policy.n_exec_log = []
    policy.plan_records = []  # E40: (env index, step 번호, 고른 계획의 예측 z^c, goal z^c, 고른 action)
    policy.oob_log = []  # E41: (첫 block의 범위 밖 비율, 전체 비율, 계획 수)

    def one_sample(info_dict):
        """solver가 받은 info를 후보 1개짜리 모양 (B, 1, ...)으로 바꾼다."""
        infos = {}
        for k, v in info_dict.items():
            if torch.is_tensor(v):
                v = v.to(device=solver.device, dtype=solver.dtype if v.is_floating_point() else None).unsqueeze(1)
            elif isinstance(v, np.ndarray):
                v = v[:, None]
            infos[k] = v
        return infos

    def plan_pred(info_dict, actions):
        """고른 action 열을 한 번 더 rollout해서 예측 z^c (B, T, d_c)와 goal z^c (B, d_c)를 낸다."""
        infos = one_sample(info_dict)
        with torch.no_grad():
            model.get_cost(infos, actions[:, None].to(device=solver.device, dtype=solver.dtype))
        d_c = getattr(model, 'dim_c', infos['predicted_emb'].size(-1))
        ge = infos['goal_emb']
        ge = ge[:, -1] if ge.ndim == 3 else ge[:, 0, -1]
        return infos['predicted_emb'][:, 0, :, :d_c].float().cpu().numpy(), ge[:, :d_c].float().cpu().numpy()

    def plan_alpha(info_dict, actions):
        pred = model.predictor
        alphas, orig = [], pred.forward

        def forward_rec(emb, act_emb):
            out = orig(emb, act_emb)
            alphas.append(pred.last_alpha[:, -1, 0].float())
            return out

        infos = one_sample(info_dict)
        pred.forward = forward_rec
        try:
            with torch.no_grad():
                model.get_cost(infos, actions[:, None].to(device=solver.device, dtype=solver.dtype))
        finally:
            del pred.forward
        return torch.stack(alphas, dim=-1).cpu()  # (B, K)

    def solve(info_dict, init_action=None):
        out = orig_solve(info_dict, init_action)
        B, K = out['actions'].shape[:2]
        policy.n_plans += B
        if plan_log:
            envs = list(getattr(policy, '_plan_envs', []))
            assert len(envs) == B, (len(envs), B)
            pred_c, goal_c = plan_pred(info_dict, out['actions'])
            acts = out['actions'].float().cpu()  # (B, K, block*A) 정규화 공간, 자르기 전 (E41)
            policy.plan_records.append((np.array(envs), np.array(policy._plan_t), pred_c, goal_c, acts.numpy()))
        if bounds is not None:  # E41/E44: 고른 action 중 env 범위 밖 비율 (plan_log와 무관하게 센다)
            lo, hi = bounds
            acts = out['actions'].float().cpu()
            o = (acts < lo) | (acts > hi)
            policy.oob_log.append((o[:, 0].float().mean().item(), o.float().mean().item(), len(acts)))
        n_exec = torch.full((B,), K, dtype=torch.long)
        if tau is not None:
            a = plan_alpha(info_dict, out['actions'])[:, :K]
            over = a > tau
            first = torch.where(over.any(1), over.float().argmax(1), torch.full((B,), K - 1))
            n_exec = first + 1
        elif rand_p is not None:  # 대조군: block마다 확률 rand_p로 끊는다 (gate와 같은 계획 횟수를 무작위 시점에)
            over = torch.as_tensor(rng.random((B, K)) < rand_p)
            first = torch.where(over.any(1), over.float().argmax(1), torch.full((B,), K - 1))
            n_exec = first + 1
        policy._pending_exec = n_exec.tolist()
        policy.n_exec_log += policy._pending_exec
        return out

    solver.solve = solve


def make_gate_replan_policy(base_cls):
    """solver가 정한 실행 block 수(_pending_exec)만큼만 action buffer를 남긴다 (E37)."""

    class GateReplanPolicy(base_cls):
        def get_action(self, info_dict, **kwargs):
            self._pending_exec = None
            n = len(self._action_buffer)
            flush = info_dict.get('_needs_flush')
            flush = np.zeros(n, bool) if flush is None else np.asarray(flush, bool)
            term = info_dict.get('terminated')
            dead = np.zeros(n, bool) if term is None else np.asarray(term, bool)
            # WorldModelPolicy.get_action과 같은 규칙: buffer가 비었거나 flush된, 살아 있는 env만 다시 계획한다
            fresh = [i for i in range(n) if (len(self._action_buffer[i]) == 0 or flush[i]) and not dead[i]]
            action = super().get_action(info_dict, **kwargs)
            if self._pending_exec is not None:
                assert len(fresh) == len(self._pending_exec), (len(fresh), len(self._pending_exec))
                blk = self.cfg.action_block
                for i, k in zip(fresh, self._pending_exec):
                    while len(self._action_buffer[i]) > k * blk - 1:
                        self._action_buffer[i].pop()
            return action

    return GateReplanPolicy


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
    # E57: planning 6개를 한 세션에서 동시에 돌리면 프로세스마다 torch thread가 수십 개라 CPU 12개에서 서로 밀린다
    # (load 32). 기본 CPU thread를 2개로 둔다. 혼자 돌릴 때는 IAWM_EVAL_THREADS로 늘린다(0이면 torch 기본값).
    _nt = int(os.environ.get("IAWM_EVAL_THREADS", "2"))
    if _nt > 0:
        torch.set_num_threads(_nt)
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

    if str(policy).startswith("oracle_"):  # E28: 정답 simulator planner (src/iawm/oracle_planner.py)
        from iawm.oracle_planner import OraclePlanner
        pc = cfg.plan_config
        policy = OraclePlanner(
            mode=str(policy).removeprefix("oracle_"),
            horizon=pc.horizon * pc.action_block, receding=pc.receding_horizon * pc.action_block,
            action_block=pc.action_block, samples=cfg.solver.num_samples, iters=cfg.solver.n_steps,
            topk=cfg.solver.topk, seed=cfg.seed, contact_cost=float(cfg.get("contact_cost", 1.0)),
            replan=str(cfg.get("replan", "none")),
        )
    elif policy != "random":
        model = swm.wm.utils.load_pretrained(cfg.policy)
        model = model.to("cuda")
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        if cfg.get("cost") is not None:  # CF-JEPA planning cost 바꾸기: sum_l2(논문 Eq. 7) | last_mse
            model.cost = cfg.cost
        if cfg.get("gate_mode") is not None:  # gate 모델의 α 바꾸기: learned | always | off (E12, off는 CF-JEPA 부분만)
            model.predictor.gate_mode = cfg.gate_mode
        if cfg.get("plan_combine") is not None:  # E34: gate on/off rollout cost를 max|mean으로 합친다
            model.plan_combine = str(cfg.plan_combine)
        if cfg.get("pos_probe") is not None:  # E40: cost=agent_px가 쓰는 위치 probe
            from iawm import pos_probe
            model.pos_probe = pos_probe.load(
                Path(swm.data.utils.get_cache_dir(sub_folder="checkpoints"), str(cfg.pos_probe)), "cuda")
        if cfg.get("contact_cost") is not None:  # E29: 상상 α로 충돌 회피 cost (iawm.cfjepa_gate.CFJEPA.get_cost)
            model.contact_cost = float(cfg.contact_cost)
        bounds = action_bounds(process, int(cfg.plan_config.action_block))
        if bool(cfg.get("action_clip", False)):  # E41
            clip_model_actions(model, *bounds)
            print(f"[action clip] normalized bounds lo {bounds[0][:2].tolist()} hi {bounds[1][:2].tolist()}", flush=True)
        config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        plan_history = int(cfg.get("plan_history", 1))
        obs_fix = bool(cfg.get("obs_fix", False))
        plan_log = cfg.get("plan_log") is not None
        policy_cls = swm.policy.WorldModelPolicy
        pr = model.predictor
        plan_oracle = (getattr(pr, "int_oracle_dim", 0) > 0 and getattr(pr, "int_oracle_key", "") == "obs_oracle"
                       and getattr(pr, "gate_mode", "off") != "off")
        est = None
        if (getattr(pr, "int_oracle_dim", 0) > 0 and getattr(pr, "int_oracle_key", "") == "obs_est"
                and getattr(pr, "gate_mode", "off") != "off"):  # E56: 탐지기 입력 모델
            from iawm import obs_detector as od
            assert cfg.get("obs_detector"), "탐지기 입력 모델은 +obs_detector=<checkpoint>가 필요하다"
            assert plan_history == pr.num_frames, "탐지기 입력 모델은 +plan_history가 모델 history 길이와 같아야 한다"
            dev = next(model.parameters()).device
            det, fc, meta = od.load(Path(swm.data.utils.get_cache_dir(), "checkpoints", str(cfg.obs_detector)), dev)
            est = (det, fc, dev)
            print(f"[obs est] detector {cfg.obs_detector} val {meta.get('det_val')} horizon {cfg.plan_config.horizon}", flush=True)
        if plan_oracle:  # E39: 정답 obstacle 상태를 planning에 넣는다
            assert plan_history == pr.num_frames, "정답 obstacle 입력 모델은 +plan_history가 모델 history 길이와 같아야 한다"
            print(f"[plan oracle] obs_oracle dim {pr.int_oracle_dim} horizon {cfg.plan_config.horizon}", flush=True)
        if (getattr(pr, "int_oracle_key", "") == "forecast" and getattr(pr, "forecaster", None) is not None
                and getattr(pr, "gate_mode", "off") != "off"):  # E45: forecaster는 history frame이 모델 history 길이만큼 있어야 한다
            assert plan_history == pr.num_frames, "forecast 입력 모델은 +plan_history가 모델 history 길이와 같아야 한다"
            print(f"[plan forecast] forecaster dim {pr.forecaster.dim} history {pr.forecaster.history} "
                  f"motion {getattr(pr.forecaster, 'motion', False)}", flush=True)
        motion = bool(getattr(getattr(pr, "forecaster", None), "motion", False)) and getattr(pr, "gate_mode", "off") != "off"
        if motion:  # E50: 1 step 전 frame도 pixels와 같은 전처리를 거친다
            assert plan_history > 1, "motion 입력 모델은 +plan_history가 필요하다"
            transform["pixels_prev"] = img_transform(cfg)
        if plan_history > 1 or obs_fix or plan_log:  # E26: 첫 관측을 실제 env 화면으로, planning에 과거 frame 넣기
            policy_cls = make_history_policy(
                policy_cls, plan_history, cfg.plan_config.action_block, obs_fix,
                oracle_horizon=cfg.plan_config.horizon if plan_oracle else 0, plan_log=plan_log,
                clip_history=bool(cfg.get("action_clip", False)), motion=motion,
                oracle_noise=float(cfg.get("oracle_noise", 0.0)) if plan_oracle else 0.0, noise_seed=int(cfg.seed),
                est=est, est_horizon=int(cfg.plan_config.horizon),
            )
        replan = str(cfg.get("replan", "none"))
        if replan in ("gate", "random"):  # E37: gate α가 tau를 넘는 block까지만 실행하고 다시 계획
            policy_cls = make_gate_replan_policy(policy_cls)
        policy = policy_cls(
            solver=solver, config=config, process=process, transform=transform
        )
        wrap_solver_count_and_gate(policy, model, float(cfg.get("replan_tau", 0.5)) if replan == "gate" else None,
                                   float(cfg.get("replan_p", 0.5)) if replan == "random" else None, int(cfg.seed),
                                   plan_log=plan_log, bounds=bounds)

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
    if getattr(policy, "n_plans", None) is not None:  # E37: 계획 횟수
        print(f"[plans] total {policy.n_plans} per_episode {policy.n_plans / cfg.eval.num_eval:.2f}")
        if getattr(policy, "n_exec_log", None):
            print(f"[plans] executed blocks per plan mean {np.mean(policy.n_exec_log):.2f} hist {np.bincount(policy.n_exec_log, minlength=6)[1:].tolist()}")
    if cfg.get("plan_log") is not None and getattr(policy, "plan_records", None):  # E40: 계획 기록 저장
        recs = policy.plan_records
        n_env, t_max = len(policy.pos_log), max(len(p) for p in policy.pos_log)
        pos = np.full((n_env, t_max, 5), np.nan, dtype=np.float32)
        for i, pl in enumerate(policy.pos_log):
            pos[i, :len(pl)] = np.stack(pl)
        out_path = Path(str(cfg.plan_log))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path, env=np.concatenate([r[0] for r in recs]), t=np.concatenate([r[1] for r in recs]),
                 pred_c=np.concatenate([r[2] for r in recs]), goal_c=np.concatenate([r[3] for r in recs]), pos=pos,
                 actions=np.concatenate([r[4] for r in recs]),
                 history=int(cfg.get("plan_history", 1)), block=int(cfg.plan_config.action_block),
                 success=np.asarray(metrics.get("episode_successes", []), dtype=bool))
        print(f"[plan log] {len(recs)} solver calls, {sum(len(r[0]) for r in recs)} plans -> {out_path}", flush=True)
    if getattr(policy, "oob_log", None):  # E41: 고른 action 중 env 범위 밖(env가 자르는) 비율
        ol = np.array(policy.oob_log)
        w = ol[:, 2] / ol[:, 2].sum()
        print(f"[actions] out of env bounds: first block {(ol[:, 0] * w).sum():.4f} all blocks {(ol[:, 1] * w).sum():.4f}", flush=True)
    ee = getattr(policy, "est_err", None)
    if ee and ee['n']:  # E56: 계획 시점의 탐지 오차 (env 정답과 비교)
        print(f"[obs est] plan-time pos_mae {np.mean(ee['pos']):.3f} vel_mae {np.mean(ee['vel']):.3f} "
              f"vel_p90 {np.quantile(ee['vel'], .9):.3f} miss {ee['miss'] / ee['n']:.4f} fp {ee['fp'] / ee['n']:.4f} n {ee['n']}", flush=True)
    if getattr(policy, "sim_err", None):
        err = np.array(policy.sim_err)
        print(f"[oracle] sim error mean {err.mean():.4f} p50 {np.median(err):.4f} p99 {np.quantile(err, .99):.4f} "
              f"frac>1px {(err > 1).mean():.4f} n={len(err)}")

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
