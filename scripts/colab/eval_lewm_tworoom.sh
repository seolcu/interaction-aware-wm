#!/usr/bin/env bash
# LeWM TwoRoom planning 평가 (CEM 300 samples × 30 iters, 50 episodes).
# setup_lewm.sh를 먼저 실행해야 한다. 결과: $STABLEWM_HOME/tworoom_results.txt
set -euo pipefail
export MPLBACKEND=Agg
export STABLEWM_HOME="${STABLEWM_HOME:-/content/stable-wm}"
export PATH="${VENV:-/content/venv}/bin:$PATH"
cd "${LEWM_DIR:-/content/le-wm}"
nvidia-smi --query-gpu=name --format=csv,noheader
time python eval.py --config-name=tworoom.yaml policy=lewm-tworooms "$@"
echo "EVAL DONE"
