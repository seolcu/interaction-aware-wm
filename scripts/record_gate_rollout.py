"""Collision TwoRoom planning을 돌리며 매 step의 상태를 기록하고, gate α_t를 계산해 GIF로 그린다 (발표 자료용).

`scripts/eval.py --config-name=collision_tworoom`과 같은 설정(CEM 300×30, horizon 5, receding 5,
시작 상태와 goal은 Collision data에서 뽑음)으로 planning한다. 다른 점은 policy를 한 겹 감싸서,
매 step 실행 직전의 화면, agent·obstacle·goal 위치, 직전 step의 contact, 실행한 action을 기록하는 것뿐이다.
기록은 planning에 영향을 주지 않는다.

α_t 계산 (planning이 끝난 뒤, 실제로 지나온 궤적에서):
- 학습 때와 같은 입력을 만든다. frame t-10, t-5, t와 각 frame에서 시작하는 action 5 step(10차원, data의
  통계로 z-score)이다. gate 모델의 predictor를 한 번 돌리고 마지막 position의 α를 쓴다.
- 그래서 α_t는 "frame t에서 다음 5 step(transition t -> t+5) 동안 obstacle의 영향이 있는가"에 대한
  gate 출력이다. analyze_gate.py의 AUROC와 같은 정의다. t < 10과 마지막 4 step은 계산하지 않는다.
- planner가 상상 rollout 안에서 쓰는 α(history 1 frame)와는 다르다. 이 GIF는 "실제 상황에서 gate가 언제 켜지는가"를 보여 준다.

출력 (--out, 기본 $STABLEWM_HOME/results/gate_rollout_k<obstacle 수>_s<seed>/):
- steps.csv: env, t, agent·goal·obstacle 위치, contact(그 step), contact_next(다음 5 step), alpha
- summary.txt: success rate, AUROC(α -> contact_next), contact별 α 평균
- gate_env<i>.gif, gate_env<i>.png: contact가 있었던 episode 중 --gifs개 (png는 마지막 frame, 전체 timeline)

사용:
    python scripts/record_gate_rollout.py --ckpt e12_sup/weights_epoch_8.pt
    python scripts/record_gate_rollout.py --ckpt e12_sup/weights_epoch_8.pt --num-obstacles 3 --gifs 6
"""

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT), str(ROOT / 'scripts')]

import numpy as np
import torch

import eval as eval_script  # noqa: E402  (env 등록, img_transform, get_dataset, get_episodes_length)
from analyze_gate import to_model_pixels  # noqa: E402

FS, HIST = 5, 3


# ---------------------------------------------------------------- planning + 기록

def make_recording_policy(base_cls):
    class RecordingPolicy(base_cls):
        """WorldModelPolicy와 같다. get_action 직전 상태와 고른 action을 env별로 기록한다."""

        def set_env(self, env):
            super().set_env(env)
            n = getattr(env, 'num_envs', 1)
            self.rec = [dict(frames=[], agent=[], goal=None, obstacles=[], contact_prev=[], actions=[])
                        for _ in range(n)]
            self.finished = np.zeros(n, dtype=bool)

        def snapshot(self, i):
            env = self.env.envs[i].unwrapped
            r = self.rec[i]
            r['frames'].append(np.asarray(env.render()).copy())
            r['agent'].append(env.agent_position.detach().cpu().numpy().copy())
            r['goal'] = env.target_position.detach().cpu().numpy().copy()
            r['obstacles'].append(env.obstacle_pos.detach().cpu().numpy().copy())
            r['contact_prev'].append(bool(env.contact))

        def get_action(self, info_dict, **kwargs):
            n = self.env.num_envs
            term = info_dict.get('terminated')
            dead = np.zeros(n, dtype=bool) if term is None else np.asarray(term, dtype=bool).reshape(n, -1).any(1)
            for i in range(n):
                if not self.finished[i]:
                    self.snapshot(i)
                    if dead[i]:
                        self.finished[i] = True
            action = super().get_action(info_dict, **kwargs)
            a = np.asarray(action).reshape(n, -1)
            for i in range(n):
                if not self.finished[i]:
                    self.rec[i]['actions'].append(a[i].copy())
            return action

        def finalize(self):
            for i in range(len(self.rec)):
                if not self.finished[i]:
                    self.snapshot(i)
                    self.finished[i] = True

    return RecordingPolicy


def run_planning(cfg, device):
    import hydra
    import stable_worldmodel as swm
    from omegaconf import OmegaConf

    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))
    transform = {'pixels': eval_script.img_transform(cfg), 'goal': eval_script.img_transform(cfg)}
    dataset = eval_script.get_dataset(cfg, cfg.eval.dataset_name)
    col_name = 'episode_idx' if 'episode_idx' in dataset.column_names else 'ep_idx'
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)

    from sklearn import preprocessing
    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col == 'pixels':
            continue
        proc = preprocessing.StandardScaler()
        d = dataset.get_col_data(col)
        proc.fit(d[~np.isnan(d).any(axis=1)])
        process[col] = proc
        if col != 'action':
            process[f'goal_{col}'] = proc

    model = swm.wm.utils.load_pretrained(cfg.policy).to(device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    if cfg.get('gate_mode') is not None:
        model.predictor.gate_mode = cfg.gate_mode
    solver = hydra.utils.instantiate(cfg.solver, model=model)
    Policy = make_recording_policy(swm.policy.WorldModelPolicy)
    policy = Policy(solver=solver, config=swm.PlanConfig(**cfg.plan_config), process=process, transform=transform)

    # 시작 위치와 goal 고르기: eval.py와 같은 방식, 같은 seed면 같은 시작점이다.
    episode_len = eval_script.get_episodes_length(dataset, ep_indices)
    max_start = dict(zip(ep_indices, episode_len - cfg.eval.goal_offset_steps - 1))
    max_start_per_row = np.array([max_start[e] for e in dataset.get_col_data(col_name)])
    valid = np.nonzero(dataset.get_col_data('step_idx') <= max_start_per_row)[0]
    g = np.random.default_rng(cfg.seed)
    rows = np.sort(valid[g.choice(len(valid) - 1, size=cfg.eval.num_eval, replace=False)])
    eval_eps = dataset.get_row_data(rows)[col_name]
    eval_start = dataset.get_row_data(rows)['step_idx']

    world.set_policy(policy)
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=eval_start.tolist(),
        goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_eps.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get('callables'), resolve=True),
    )
    policy.finalize()
    return model, process['action'], policy.rec, metrics


# ---------------------------------------------------------------- α 계산

@torch.no_grad()
def compute_alpha(model, act_scaler, rec, device, batch=64):
    """env 하나의 기록에서 t마다 α_t를 계산한다. 반환: (T+1,) 배열, 계산하지 않은 t는 NaN."""
    frames = np.stack(rec['frames'])               # (T+1, H, W, 3), frame t = action t 직전
    acts = np.stack(rec['actions']) if rec['actions'] else np.zeros((0, 2))   # (T, 2)
    T = len(acts)
    alpha = np.full(len(frames), np.nan, dtype=np.float32)
    if not hasattr(model.predictor, 'last_alpha'):
        return alpha
    acts_n = act_scaler.transform(acts).astype(np.float32) if T else acts
    ts = np.arange(2 * FS, T - FS + 1)              # 필요한 action: t-10 .. t+4
    for i in range(0, len(ts), batch):
        tb = ts[i:i + batch]
        fidx = tb[:, None] + FS * (np.arange(HIST) - (HIST - 1))[None]          # t-10, t-5, t
        aidx = fidx[:, :, None] + np.arange(FS)[None, None]                       # 각 frame부터 5 step
        px = to_model_pixels(frames[fidx], device)
        a = torch.from_numpy(acts_n[aidx].reshape(len(tb), HIST, -1)).to(device)
        out = model.encode({'pixels': px, 'action': a})
        model.predict(out['emb'], out['act_emb'])
        alpha[tb] = model.predictor.last_alpha[:, -1, 0].float().cpu().numpy()
    return alpha


# ---------------------------------------------------------------- 그리기

BLUE, ORANGE, GREEN, INK, MUTED, GRID = (42, 120, 214), (235, 104, 52), (0, 150, 0), (11, 11, 11), (82, 81, 78), (215, 214, 210)


def load_font(size):
    from PIL import ImageFont
    try:
        import matplotlib
        return ImageFont.truetype(str(Path(matplotlib.get_data_path()) / 'fonts/ttf/DejaVuSans.ttf'), size)
    except Exception:
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()


def render_gif(rec, alpha, contact, out_png, out_gif, success, title, scale=2, duration=90):
    from PIL import Image, ImageDraw

    S = 224 * scale
    head, foot = 64, 196
    f_big, f_mid, f_small = load_font(22), load_font(16), load_font(13)
    T1 = len(rec['frames'])
    goal = rec['goal'] * scale

    # timeline panel geometry
    px0, px1 = 46, S - 14
    py0, py1 = S + head + 26, S + head + foot - 52
    xs = lambda t: px0 + (px1 - px0) * t / max(T1 - 1, 1)   # noqa: E731
    ys = lambda v: py1 - (py1 - py0) * v                     # noqa: E731

    def frame(t, final=False):
        img = Image.new('RGB', (S, S + head + foot), (252, 252, 251))
        scene = Image.fromarray(rec['frames'][t]).resize((S, S), Image.BICUBIC)
        img.paste(scene, (0, head))
        d = ImageDraw.Draw(img, 'RGBA')
        oy = head
        a = alpha[t]
        on = bool(np.isfinite(a) and a > 0.5)
        ag = rec['agent'][t] * scale
        # gate가 켜졌으면 obstacle -> agent 영향선(u -> c). 굵기와 진하기는 α에 비례한다.
        if np.isfinite(a):
            for ob in rec['obstacles'][t]:
                ob = ob * scale
                if np.linalg.norm(ob - ag) < 140 * scale / 2:
                    w = int(2 + 8 * a)
                    d.line([ob[0], ob[1] + oy, ag[0], ag[1] + oy], fill=(*ORANGE, int(40 + 200 * a)), width=w)
        # goal X (env 화면에는 goal이 그려지지 않는다. 시각화 전용 표시)
        gx, gy = goal[0], goal[1] + oy
        d.line([gx - 11, gy - 11, gx + 11, gy + 11], fill=(*GREEN, 255), width=4)
        d.line([gx - 11, gy + 11, gx + 11, gy - 11], fill=(*GREEN, 255), width=4)
        if contact[t]:
            d.rectangle([0, oy, S - 1, oy + S - 1], outline=(*ORANGE, 255), width=6)
        # header
        d.text((14, 8), title, font=f_mid, fill=MUTED)
        d.text((14, 32), f't = {t:2d}', font=f_big, fill=INK)
        if final:
            txt = 'goal reached' if success else 'goal not reached'
        else:
            txt = 'α_t = —' if not np.isfinite(a) else f'α_t = {a:.2f}'
        d.text((120, 32), txt, font=f_big, fill=INK)
        if not final:
            pill = 'GATE ON' if on else ('GATE OFF' if np.isfinite(a) else 'GATE —')
            pw = 118
            d.rounded_rectangle([S - pw - 14, 30, S - 14, 58], radius=12,
                                fill=(*ORANGE, 255) if on else (230, 229, 225, 255))
            d.text((S - 14 - pw / 2, 44), pill, font=f_mid, fill=(255, 255, 255) if on else MUTED, anchor='mm')
        # timeline: contact 구간(주황 띠), α 곡선(파랑), 0.5 기준선, 현재 t
        d.text((14, S + head + 4), 'gate α_t (line)   contact (band)', font=f_small, fill=MUTED)
        k = 0
        while k < T1:  # 연속한 contact step은 띠 하나로 그린다
            if contact[k]:
                k0 = k
                while k + 1 < T1 and contact[k + 1]:
                    k += 1
                d.rectangle([xs(k0 - 0.5), py0, xs(k + 0.5), py1], fill=(*ORANGE, 70))
            k += 1
        for v in (0.0, 0.5, 1.0):
            d.line([px0, ys(v), px1, ys(v)], fill=GRID if v != 0.5 else (170, 169, 165), width=1)
            d.text((px0 - 8, ys(v)), f'{v:.1f}', font=f_small, fill=MUTED, anchor='rm')
        upto = T1 if final else t + 1
        pts = [(xs(k), ys(alpha[k])) for k in range(upto) if np.isfinite(alpha[k])]
        if len(pts) > 1:
            d.line(pts, fill=(*BLUE, 255), width=3, joint='curve')
        if not final:
            d.line([xs(t), py0 - 4, xs(t), py1 + 4], fill=INK, width=1)
        d.text((px0, py1 + 4), '0', font=f_small, fill=MUTED)
        d.text((px1, py1 + 4), f'{T1 - 1} step', font=f_small, fill=MUTED, anchor='ra')
        d.text((S / 2, S + head + foot - 12), 'red: agent  ·  blue: obstacle  ·  X: goal  ·  orange link: gate ON',
               font=f_small, fill=MUTED, anchor='mm')
        return img

    frames = [frame(t) for t in range(T1)]
    last = frame(T1 - 1, final=True)
    frames += [last] * 12
    frames[0].save(out_gif, save_all=True, append_images=frames[1:], duration=duration, loop=0, optimize=True)
    last.save(out_png)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='$STABLEWM_HOME/checkpoints 기준 gate 모델 checkpoint')
    ap.add_argument('--num-eval', type=int, default=16, help='planning episode 수 (병렬 env 수)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--num-obstacles', type=int, default=1)
    ap.add_argument('--gifs', type=int, default=4, help='GIF로 그릴 episode 수 (contact가 있었던 것 중)')
    ap.add_argument('--out', default=None)
    ap.add_argument('--device', default=None)
    ap.add_argument('overrides', nargs='*', help='eval config hydra override (예: solver.num_samples=100)')
    args = ap.parse_args()

    from hydra import compose, initialize_config_dir
    from sklearn.metrics import roc_auc_score
    import stable_worldmodel as swm

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    with initialize_config_dir(config_dir=str(ROOT / 'configs/eval'), version_base=None):
        cfg = compose(config_name='collision_tworoom', overrides=[
            f'policy={args.ckpt}', f'seed={args.seed}', f'eval.num_eval={args.num_eval}',
            f'world.num_obstacles={args.num_obstacles}', '+world.obstacle_info=first',
            f'solver.device={device}', *args.overrides,
        ])
    out = Path(args.out or Path(swm.data.utils.get_cache_dir(), 'results',
                                f'gate_rollout_k{args.num_obstacles}_s{args.seed}'))
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    model, act_scaler, recs, metrics = run_planning(cfg, device)
    succ = np.asarray(metrics['episode_successes'], dtype=bool)

    rows, all_a, all_c = [], [], []
    per_env = []
    for i, rec in enumerate(recs):
        T1 = len(rec['frames'])
        alpha = compute_alpha(model, act_scaler, rec, device)
        # contact[k]: frame k에 보이는 상태를 만든 step(k-1 -> k)에서 contact가 있었는가
        contact = np.array(rec['contact_prev'], dtype=bool)
        contact[0] = False
        # contact_next[t]: frame t 이후 5 step(t -> t+5) 안에 contact가 있는가 (α_t의 정답과 같은 정의)
        cn = np.array([contact[t + 1:t + FS + 1].any() for t in range(T1)])
        for t in range(T1):
            obs = rec['obstacles'][t]
            rows.append([i, t, *np.round(rec['agent'][t], 2), *np.round(rec['goal'], 2),
                         len(obs), *(np.round(obs[0], 2) if len(obs) else (np.nan, np.nan)),
                         int(contact[t]), int(cn[t]), alpha[t]])
        m = np.isfinite(alpha)
        all_a.append(alpha[m]), all_c.append(cn[m])
        per_env.append((i, alpha, contact, cn))

    with open(out / 'steps.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['env', 't', 'agent_x', 'agent_y', 'goal_x', 'goal_y', 'num_obstacles',
                    'obstacle0_x', 'obstacle0_y', 'contact', 'contact_next', 'alpha'])
        w.writerows(rows)

    a, c = np.concatenate(all_a), np.concatenate(all_c)
    lines = [
        f'ckpt={args.ckpt} num_obstacles={args.num_obstacles} seed={args.seed} episodes={len(recs)}',
        f'success_rate={metrics["success_rate"]:.1f}',
        f'steps with alpha: {len(a)}  contact_next ratio: {c.mean():.4f}',
    ]
    if c.any() and (~c).any():
        lines += [f'AUROC(alpha -> contact_next) = {roc_auc_score(c, a):.4f}',
                  f'alpha mean non-contact / contact = {a[~c].mean():.4f} / {a[c].mean():.4f}',
                  f'open ratio(alpha>0.5) non-contact / contact = {(a[~c] > .5).mean():.4f} / {(a[c] > .5).mean():.4f}']
    (out / 'summary.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))

    # GIF: contact가 있고 α가 계산된 episode. 성공한 것, contact가 많은 것 순.
    cand = [e for e in per_env if e[2].any() and np.isfinite(e[1]).any()]
    cand.sort(key=lambda e: (not succ[e[0]], -int(e[2].sum())))
    for i, alpha, contact, _ in cand[:args.gifs]:
        title = f'Collision TwoRoom · obstacle {args.num_obstacles} · episode {i}'
        render_gif(recs[i], alpha, contact, out / f'gate_env{i}.png', out / f'gate_env{i}.gif', bool(succ[i]), title)
        print(f'saved {out}/gate_env{i}.gif (success={bool(succ[i])}, contact steps={int(contact.sum())})')


if __name__ == '__main__':
    main()
