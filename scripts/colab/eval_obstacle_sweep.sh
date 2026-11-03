#!/usr/bin/env bash
# Obstacle 개수별 Collision TwoRoom planning 성공률 (발표 자료용, 재학습 없음).
# E14와 같은 설정이다: --config-name=collision_tworoom, CEM 300×30, 50 episodes, +world.obstacle_info=first,
# 시작 상태와 goal은 obstacle 1개 data($STABLEWM_HOME/datasets/collision_tworoom.h5)에서 뽑는다.
#
# 사용 (setup_lewm.sh, data, checkpoint가 준비된 Colab VM에서):
#   bash scripts/colab/eval_obstacle_sweep.sh                      # obstacle 2, 4, 5 × seed 42, 43 × 모델 2개
#   OBSTACLES="1 2 3 4 5" bash scripts/colab/eval_obstacle_sweep.sh   # E13·E14 기록값 대신 전부 새로 잰다
#   MODELS="cfjepa=cfjepa_collision_n2/weights_epoch_10.pt" bash scripts/colab/eval_obstacle_sweep.sh
# 이미 끝난 조합은 건너뛴다(세션이 끊겨도 다시 실행하면 이어서 한다).
# 결과: $STABLEWM_HOME/results/sweep_<모델>_k<개수>_s<seed>/, 요약 $STABLEWM_HOME/results/obstacle_sweep_summary.txt
# 그래프: python scripts/plot_obstacle_sweep.py
set -uo pipefail
export MPLBACKEND=Agg
export STABLEWM_HOME="${STABLEWM_HOME:-/content/stable-wm}"
export PATH="${VENV:-/content/venv}/bin:$PATH"
cd "$(dirname "$0")/../.."

OBSTACLES="${OBSTACLES:-2 4 5}"
SEEDS="${SEEDS:-42 43}"
MODELS="${MODELS:-cfjepa=cfjepa_collision_n2/weights_epoch_10.pt gate=e12_sup/weights_epoch_8.pt}"
RES="$STABLEWM_HOME/results"
SUMMARY="$RES/obstacle_sweep_summary.txt"
mkdir -p "$RES"

for k in $OBSTACLES; do
  for s in $SEEDS; do
    for m in $MODELS; do
      tag="${m%%=*}"; ckpt="${m#*=}"; name="sweep_${tag}_k${k}_s${s}"
      out="$RES/$name/$name.txt"
      if [ -f "$out" ] && grep -q "success_rate" "$out"; then
        echo "skip $name (이미 있음)"; continue
      fi
      echo "[$(date +%H:%M)] run $name"
      python scripts/eval.py --config-name=collision_tworoom policy="$ckpt" seed="$s" \
        world.num_obstacles="$k" +world.obstacle_info=first output.filename="$name.txt" \
        > "$RES/$name.log" 2>&1
      sr=$(grep -o "'success_rate': [0-9.]*" "$out" 2>/dev/null | tail -1 | awk '{print $2}')
      echo "$name ${sr:-FAILED}" | tee -a "$SUMMARY"
    done
  done
done
echo "SWEEP DONE"
