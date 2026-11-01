"""평가 설정 grid를 튜닝용 시작점에서 돌리고 요약한다 (E6).

보고용 시작점(eval seed 42)과 겹치지 않게 seed 1~4(각 50 episode, 합 200)를 쓴다.
사용: python scripts/colab/tune_eval.py <ckpt_dir> <tag>=<epoch>:<cost> ...
  python scripts/colab/tune_eval.py cfjepa_tworoom_n2 e10_sum=10:sum_l2 e10_last=10:last_mse
결과: /content/tune_<tag>.log 와 마지막 요약 표
"""
import re
import subprocess
import sys

ckpt_dir, specs = sys.argv[1], sys.argv[2:]
SEEDS = [1, 2, 3, 4]
summary = {}
for spec in specs:
    tag, rest = spec.split('=')
    epoch, cost = rest.split(':')
    rates = []
    for seed in SEEDS:
        cmd = (f'bash scripts/colab/eval_cfjepa.sh {ckpt_dir}/weights_epoch_{epoch}.pt 2 '
               f'seed={seed} +cost={cost} output.filename=tune_{tag}_s{seed}.txt')
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd='/content/iawm')
        open(f'/content/tune_{tag}.log', 'a').write(out.stdout[-3000:] + out.stderr[-3000:])
        m = re.search(r"'success_rate': ([\d.]+)", out.stdout)
        rates.append(float(m.group(1)) if m else float('nan'))
        print(tag, 'seed', seed, rates[-1], flush=True)
    summary[tag] = rates
print('==== SUMMARY (N=2, 200 tuning episodes) ====')
for tag, r in summary.items():
    print(f'{tag}: mean {sum(r) / len(r) / 100:.3f}  per-seed {r}')
print('TUNE DONE')
