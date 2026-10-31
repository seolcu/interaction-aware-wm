#!/usr/bin/env bash
# Planning 평가 (CEM 300 samples × 30 iters, 50 episodes).
# 사용: bash eval_cfjepa.sh <checkpoint> <num_distractors> [hydra override ...]
#   bash eval_cfjepa.sh cfjepa_tworoom_n2/weights_epoch_10.pt 2
# 결과: $STABLEWM_HOME/results/<output.filename 확장자 제외>/
set -euo pipefail
export MPLBACKEND=Agg
export STABLEWM_HOME="${STABLEWM_HOME:-/content/stable-wm}"
export PATH="${VENV:-/content/venv}/bin:$PATH"
cd "$(dirname "$0")/../.."
CKPT="$1"; N="$2"; shift 2
NAME="$(echo "$CKPT" | tr '/' '_' | sed 's/\.pt$//')_n${N}"
time python scripts/eval.py policy="$CKPT" world.num_distractors="$N" output.filename="${NAME}.txt" "$@"
echo "EVAL DONE"
