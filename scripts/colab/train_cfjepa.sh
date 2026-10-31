#!/usr/bin/env bash
# Colab VM에서 world model을 학습한다. setup_lewm.sh를 먼저 실행해야 한다.
# 사용: bash train_cfjepa.sh [hydra override ...]
#   bash train_cfjepa.sh                                   # CF-JEPA, TwoRoom, N=2, 10 epochs
#   bash train_cfjepa.sh model=lewm output_model_name=lewm_tworoom_n2
# Checkpoint: $STABLEWM_HOME/checkpoints/<output_model_name>/weights_epoch_<k>.pt (매 epoch)
set -euo pipefail
export MPLBACKEND=Agg
export STABLEWM_HOME="${STABLEWM_HOME:-/content/stable-wm}"
export PATH="${VENV:-/content/venv}/bin:$PATH"
cd "$(dirname "$0")/../.."
nvidia-smi --query-gpu=name --format=csv,noheader
time python scripts/train.py "$@"
echo "TRAIN DONE"
