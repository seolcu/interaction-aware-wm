"""Obstacle 개수별 planning 성공률 그래프 (발표 자료용).

`scripts/colab/eval_obstacle_sweep.sh`의 결과(`$STABLEWM_HOME/results/sweep_<모델>_k<개수>_s<seed>/*.txt`)를 읽는다.
새로 재지 않은 개수는 docs/experiments.md에 기록된 값으로 채운다(--no-records로 끈다).
- obstacle 1개: E13 (seed 42·43·44, 150 episodes)
- obstacle 3개: E14 `c3` (seed 42·43, 100 episodes)
같은 개수를 새로 쟀으면 새 값을 쓴다. 기록값은 그래프에 속이 빈 marker로 표시한다.

오차 막대는 성공률의 binomial 표준오차 sqrt(p(1-p)/n)다(n = episode 수).

출력: obstacle_sweep.png, obstacle_sweep.svg, obstacle_sweep.csv (--out 폴더, 기본 $STABLEWM_HOME/results)

사용:
    python scripts/plot_obstacle_sweep.py
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

EPISODES_PER_RUN = 50
# docs/experiments.md E13, E14에 적힌 값: (모델, obstacle 개수) -> (성공률 %, episode 수, 출처)
RECORDS = {
    ('cfjepa', 1): (91.3, 150, 'E13'),
    ('gate', 1): (91.3, 150, 'E13'),
    ('cfjepa', 3): (73.0, 100, 'E14'),
    ('gate', 3): (71.0, 100, 'E14'),
}
LABELS = {'cfjepa': 'CF-JEPA (no gate)', 'gate': 'CF-JEPA + interaction gate'}
COLORS = {'cfjepa': '#2a78d6', 'gate': '#eb6834'}  # reference palette slot 1, 2 (validator PASS)


def collect(results_dir: Path):
    runs = defaultdict(list)  # (tag, k) -> [success %]
    for f in sorted(results_dir.glob('sweep_*_k*_s*/sweep_*_k*_s*.txt')):
        m = re.match(r'sweep_(.+)_k(\d+)_s(\d+)$', f.stem)
        hits = re.findall(r"'success_rate': ([0-9.]+)", f.read_text())
        if m and hits:
            runs[(m.group(1), int(m.group(2)))].append(float(hits[-1]))
    return runs


def main():
    import stable_worldmodel as swm

    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=None, help='기본 $STABLEWM_HOME/results')
    ap.add_argument('--out', default=None)
    ap.add_argument('--no-records', action='store_true', help='E13·E14 기록값을 쓰지 않는다')
    args = ap.parse_args()
    res = Path(args.results or Path(swm.data.utils.get_cache_dir(), 'results'))
    out = Path(args.out or res)
    out.mkdir(parents=True, exist_ok=True)

    runs = collect(res)
    table = {}  # (tag, k) -> (p, n, source)
    for (tag, k), v in runs.items():
        table[(tag, k)] = (float(np.mean(v)), EPISODES_PER_RUN * len(v), f'measured ({len(v)} seeds)')
    if not args.no_records:
        for key, val in RECORDS.items():
            table.setdefault(key, val)

    tags = [t for t in ('cfjepa', 'gate') if any(k[0] == t for k in table)]
    tags += sorted({k[0] for k in table} - set(tags))

    with open(out / 'obstacle_sweep.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'num_obstacles', 'success_rate', 'episodes', 'std_err', 'source'])
        for tag in tags:
            for k in sorted(kk for t, kk in table if t == tag):
                p, n, src = table[(tag, k)]
                se = 100 * np.sqrt((p / 100) * (1 - p / 100) / n)
                w.writerow([tag, k, round(p, 1), n, round(se, 1), src])
                print(f'{tag:8s} k={k}  {p:5.1f}%  ± {se:4.1f}  (n={n}, {src})')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ink, ink2, grid, surface = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=200)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    ends = []  # (tag, x, y) 끝 값 라벨
    for i, tag in enumerate(tags):
        ks = sorted(kk for t, kk in table if t == tag)
        p = np.array([table[(tag, k)][0] for k in ks])
        n = np.array([table[(tag, k)][1] for k in ks])
        se = 100 * np.sqrt((p / 100) * (1 - p / 100) / n)
        rec = np.array(['measured' not in table[(tag, k)][2] for k in ks])
        c = COLORS.get(tag, '#6250d6')
        x = np.array(ks) + (i - (len(tags) - 1) / 2) * 0.06  # 오차 막대가 겹치지 않게 살짝 비킨다
        ax.errorbar(x, p, yerr=se, color=c, lw=2, capsize=0, elinewidth=1.2, zorder=3, label=LABELS.get(tag, tag))
        ax.scatter(x[~rec], p[~rec], s=64, color=c, edgecolor=surface, linewidth=2, zorder=4)
        ax.scatter(x[rec], p[rec], s=64, facecolor=surface, edgecolor=c, linewidth=2, zorder=4)
        ends.append((x[-1], p[-1]))
    # 끝 값 라벨: 두 값이 가까우면 위아래로 벌린다
    order = np.argsort([e[1] for e in ends])
    ys_lab = [ends[j][1] for j in order]
    for a in range(1, len(ys_lab)):
        ys_lab[a] = max(ys_lab[a], ys_lab[a - 1] + 6)
    for j, yl in zip(order, ys_lab):
        ax.annotate(f'{ends[j][1]:.0f}%', (ends[j][0], ends[j][1]), xytext=(max(e[0] for e in ends) + 0.18, yl),
                    textcoords='data', va='center', fontsize=10, color=ink)
    all_k = sorted({k for _, k in table})
    ax.set_xticks(all_k)
    ax.set_xlim(min(all_k) - 0.4, max(all_k) + 0.6)
    ax.set_ylim(0, 100)
    ax.set_xlabel('Number of moving obstacles (evaluation)', color=ink2, fontsize=10)
    ax.set_ylabel('Planning success rate (%)', color=ink2, fontsize=10)
    ax.set_title('Collision TwoRoom: planning success vs. number of obstacles', loc='left', color=ink, fontsize=12)
    ax.grid(axis='y', color=grid, lw=0.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(grid)
    ax.tick_params(colors=ink2, labelsize=9)
    ax.legend(loc='lower left', frameon=False, fontsize=9, labelcolor=ink)
    note = ('Models trained with 1 obstacle. 50 episodes per seed; error bars = binomial s.e.\n'
            'Hollow markers: values recorded in E13 (1 obstacle) / E14 (3 obstacles).')
    fig.text(0.01, 0.01, note, fontsize=7.5, color=ink2, va='bottom')
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out / 'obstacle_sweep.png', facecolor=surface)
    fig.savefig(out / 'obstacle_sweep.svg', facecolor=surface)
    print(f'saved {out}/obstacle_sweep.png')


if __name__ == '__main__':
    main()
