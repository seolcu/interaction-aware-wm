#!/usr/bin/env bash
# Colab VM에서 LeWM TwoRoom baseline 환경을 준비한다.
# 사용: STABLEWM_HOME=/content/stable-wm bash setup_lewm.sh
# 이후 python 실행은 /content/venv/bin/python 을 쓴다.
# TASK=pusht 이면 TwoRoom 대신 PushT data와 공개 LeWM PushT checkpoint를 받는다.
# SKIP_DATA=1 이면 data와 checkpoint 단계(3, 4)를 건너뛴다. env만 필요할 때 쓴다.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Colab kernel의 inline backend가 venv로 상속되면 matplotlib import가 실패한다.
export MPLBACKEND=Agg
export STABLEWM_HOME="${STABLEWM_HOME:-/content/stable-wm}"
LEWM_DIR="${LEWM_DIR:-/content/le-wm}"
mkdir -p "$STABLEWM_HOME"

echo "[1/4] le-wm clone"
[ -d "$LEWM_DIR" ] || git clone -q https://github.com/lucas-maes/le-wm "$LEWM_DIR"
git -C "$LEWM_DIR" log --oneline -1

echo "[2/4] Python 3.10 venv에 stable-worldmodel 설치"
# Colab 기본 Python(3.13)에는 box2d-py, labmaze wheel이 없어서 LeWM README대로 3.10을 쓴다.
VENV="${VENV:-/content/venv}"
pip install -q uv
[ -x "$VENV/bin/python" ] || uv venv -q --python 3.10 "$VENV"
# resolver가 datasets 1.1.1(pyarrow 최신과 비호환)을 고르는 일이 있어 하한을 둔다.
uv pip install -q --python "$VENV/bin/python" "stable-worldmodel[train,env,format]" huggingface_hub "datasets>=3" "transformers>=5"
export PATH="$VENV/bin:$PATH"
python -c "import stable_worldmodel as swm, stable_pretraining; print('swm', swm.__version__ if hasattr(swm,'__version__') else 'ok')"

if [ "${SKIP_DATA:-0}" = "1" ]; then echo "SKIP_DATA=1: 3, 4단계 생략"; echo "SETUP DONE"; exit 0; fi

echo "[3/4] HF data download (checkpoint는 4단계에서 받는다), TASK=${TASK:-tworoom}"
command -v zstd >/dev/null || { apt-get -qq update || true; apt-get -qq install -y zstd >/dev/null; }
# stable-worldmodel 0.1.x는 dataset을 $STABLEWM_HOME/datasets/<name>.h5 에서 찾는다.
mkdir -p "$STABLEWM_HOME/datasets"
if [ "${TASK:-tworoom}" = "pusht" ]; then
  # PushT data는 tar가 아니라 h5 하나를 zstd로 압축한 파일이다(13GB).
  hf download quentinll/lewm-pusht --repo-type dataset --local-dir "$STABLEWM_HOME/hf_pusht_data"
  [ -f "$STABLEWM_HOME/datasets/pusht_expert_train.h5" ] || \
    zstd -d -q "$STABLEWM_HOME/hf_pusht_data/pusht_expert_train.h5.zst" -o "$STABLEWM_HOME/datasets/pusht_expert_train.h5"
  rm -f "$STABLEWM_HOME/hf_pusht_data/pusht_expert_train.h5.zst"  # 디스크 절약
  HF_MODEL=quentinll/lewm-pusht; CKPT_NAME=lewm-pusht
else
  hf download quentinll/lewm-tworooms --repo-type dataset --local-dir "$STABLEWM_HOME/hf_tworooms_data"
  if ! ls "$STABLEWM_HOME"/datasets/tworoom*.h5 >/dev/null 2>&1; then
    tar --zstd -xf "$STABLEWM_HOME"/hf_tworooms_data/*.tar.zst -C "$STABLEWM_HOME/datasets"
  fi
  HF_MODEL=quentinll/lewm-tworooms; CKPT_NAME=lewm-tworooms
fi
ls -la "$STABLEWM_HOME/datasets"; df -h "$STABLEWM_HOME" | tail -1

echo "[4/4] checkpoint key를 transformers v5 형식으로 변환"
python "$SCRIPT_DIR/fix_lewm_keys.py" "$HF_MODEL" "$CKPT_NAME"
echo "SETUP DONE"
