"""planner가 고른 계획에서 예측이 맞았는지 본다 (E40).

`scripts/eval.py +plan_log=<npz>`가 남긴 기록을 읽는다. 계획마다 고른 action 열의 예측 z^c가 있고, env step마다
실제 agent 위치가 있다. 위치 probe(iawm.pos_probe)로 예측 z^c를 px 위치로 읽어 실제와 비교한다.

E39에서 정답 obstacle 입력 모델은 수집 data의 action에서 5 block 앞 agent 위치를 6.5px 오차로 맞혔다(CF-JEPA 15.9px).
그런데 planning 성공률은 거의 같았다. CEM은 모델이 낙관적으로 틀리는 action 열을 골라낼 수 있으므로,
planner가 실제로 고른 계획에서의 오차는 다를 수 있다(arXiv 2607.10362).

episode의 첫 계획(step 0에서 세운 25 step 계획)만 본다. 평가는 이 계획을 한 번에 실행하므로 예측과 실제를 그대로
비교할 수 있다. 내는 값:
- k block 뒤 agent 위치의 예측 오차(px). contact는 그때까지 실제로 충돌이 있었던 계획이다.
- 계획이 "도착한다"고 예측했는가(예측 위치가 어느 block에서든 goal 16px 안) 대 실제 성공. 실패한 episode 중
  "예측으로는 도착"의 비율이 높으면 모델이 planner가 고른 계획에서 틀린 것이고, 낮으면 planner가 도착하는
  계획을 찾지 못한 것이다.
- 참고: goal image에서 읽은 goal 위치의 오차, 현재 frame에서 읽은 agent 위치의 오차.

사용:
    python scripts/analyze_plan_log.py --probe pos_probe_e32_cf.pt --logs /content/e40/r1x_cf_s42_planlog.npz,...
    (같은 조건의 seed 여러 개를 --logs에 주면 합쳐서 낸다)
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

import numpy as np
import torch

REACH = 16.0  # env의 성공 기준 (목표까지 16px 미만)


def one(path, probe, device):
    d = np.load(path)
    H, blk = int(d['history']), int(d['block'])
    env, t, pos, succ = d['env'], d['t'], d['pos'], d['success']
    first = t == 0
    env = env[first]
    with torch.no_grad():
        pred = probe(torch.from_numpy(d['pred_c'][first]).to(device))[..., :2].cpu().numpy()  # (P, T, 2)
        goal = probe(torch.from_numpy(d['goal_c'][first]).to(device))[..., :2].cpu().numpy()  # (P, 2)
    K = pred.shape[1] - H
    cur, fut = pred[:, H - 1], pred[:, H:]                    # 현재 frame에서 읽은 위치, k = 1..K block 뒤 예측
    start, target = pos[env, 0, :2], pos[env, 0, 2:4]
    rows = {'env': env, 'succ': succ[env] if len(succ) else np.zeros(len(env), bool),
            'goal_err': np.linalg.norm(goal - target, axis=1), 'cur_err': np.linalg.norm(cur - start, axis=1),
            'pred_goal_dist': np.linalg.norm(fut - goal[:, None], axis=2),      # (P, K) goal image 기준
            'pred_true_dist': np.linalg.norm(fut - target[:, None], axis=2)}    # (P, K) 실제 target 기준
    err = np.full((len(env), K), np.nan, dtype=np.float32)
    con = np.zeros((len(env), K), bool)
    # 성공한 env는 그 step 뒤로 위치가 멈춘다(기록은 계속된다). 멈춘 위치와 예측을 비교하면 오차가 부풀므로,
    # goal에 처음 닿은 step까지만 비교한다. (E40 첫 집계에서는 이 처리가 없어 오차가 16~36px로 나왔다.)
    dist = np.linalg.norm(pos[env, :, :2] - pos[env, :, 2:4], axis=2)       # (P, steps)
    hit = np.where((dist < REACH).any(1), (dist < REACH).argmax(1), pos.shape[1])
    rows['hit'] = hit
    for j, e in enumerate(env):
        for k in range(1, K + 1):
            s = k * blk
            if s >= pos.shape[1] or np.isnan(pos[e, s, 0]) or s > hit[j]:
                break
            # 성공하면 World가 env를 바로 reset할 수도 있다. target이 바뀌었으면 그 뒤는 다른 episode다.
            if np.abs(pos[e, 1:s + 1, 2:4] - target[j]).max() > 1e-3:
                break
            err[j, k - 1] = np.linalg.norm(fut[j, k - 1] - pos[e, s, :2])
            con[j, k - 1] = pos[e, 1:s + 1, 4].max() > 0
    rows['err'], rows['con'] = err, con
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logs', required=True, help='쉼표로 나눈 npz 경로 (같은 조건의 seed들)')
    ap.add_argument('--probe', required=True, help='$STABLEWM_HOME/checkpoints 아래 probe 파일')
    ap.add_argument('--name', default='plans')
    args = ap.parse_args()

    from iawm import pos_probe

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    path = Path(args.probe)
    if not path.exists():  # 파일 이름만 주면 $STABLEWM_HOME/checkpoints에서 찾는다
        import stable_worldmodel as swm
        path = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), args.probe)
    probe = pos_probe.load(path, device)
    parts = [one(p, probe, device) for p in args.logs.split(',') if Path(p).exists()]
    assert parts, 'no plan logs found'
    r = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    n, K = r['err'].shape
    succ = r['succ']
    print(f'{args.name}: {len(parts)} logs, {n} first plans, success {succ.mean() * 100:.1f}%')
    print(f'  probe 확인: 현재 frame의 agent 위치 오차 {r["cur_err"].mean():.2f}px, goal image에서 읽은 goal 위치 오차 {r["goal_err"].mean():.2f}px')
    print(f'  {"k":>3}{"n":>6}{"err all":>9}{"no-contact":>12}{"contact":>9}{"n contact":>11}')
    for k in range(K):
        v = ~np.isnan(r['err'][:, k])
        c = r['con'][:, k] & v
        m = lambda mask: r['err'][mask, k].mean() if mask.any() else float('nan')  # noqa: E731
        print(f'  {k + 1:>3}{v.sum():>6}{m(v):>9.2f}{m(v & ~c):>12.2f}{m(c):>9.2f}{c.sum():>11}')
    kk = K - 1
    v = ~np.isnan(r['err'][:, kk])
    c = r['con'][:, kk] & v
    reach = (r['pred_goal_dist'].min(1) < REACH)          # 계획이 도착한다고 예측했는가 (goal image 기준)
    reach_true = (r['pred_true_dist'].min(1) < REACH)     # 같은 것을 실제 target 위치 기준으로
    fail = ~succ
    print(f'  계획이 도착한다고 예측: {reach.mean() * 100:.1f}% (실제 target 기준 {reach_true.mean() * 100:.1f}%)')
    print(f'  예측 도착 & 실제 성공 {(reach & succ).sum()} / 예측 도착 & 실제 실패 {(reach & fail).sum()} / '
          f'예측 미도착 & 실제 성공 {(~reach & succ).sum()} / 예측 미도착 & 실제 실패 {(~reach & fail).sum()}')
    fr = (reach & fail).sum() / max(1, fail.sum())
    last = r['pred_goal_dist'][:, -1]
    print(f'  실패한 episode 중 "예측으로는 도착": {fr * 100:.1f}% ({(reach & fail).sum()}/{fail.sum()}). '
          f'마지막 block의 예측 goal 거리 평균: 성공 {last[succ].mean():.1f}px, 실패 {last[fail].mean() if fail.any() else float("nan"):.1f}px')
    print(f'RESULT_PLAN {args.name} n {n} success {succ.mean():.4f} err_k{K}_nc {r["err"][v & ~c, kk].mean() if (v & ~c).any() else float("nan"):.3f} '
          f'err_k{K}_c {r["err"][c, kk].mean() if c.any() else float("nan"):.3f} pred_reach {reach.mean():.4f} '
          f'fail_pred_reach {fr:.4f} n_fail {int(fail.sum())} goal_err {r["goal_err"].mean():.3f}')


if __name__ == '__main__':
    main()
