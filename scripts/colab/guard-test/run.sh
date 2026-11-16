#!/usr/bin/env bash
# colab-guard 시험: 가짜 colab 명령(fakecolab)과 임시 HOME에서 sweep, reg, hook 동작을 확인한다. CU를 쓰지 않는다.
# 사용: bash scripts/colab/guard-test/run.sh [guard 경로]   (기본: scripts/colab/colab-guard)
# colab-guard나 selfstop.py를 고치면 설치 전에 돌린다. 마지막 줄이 "RESULT pass=N fail=0"이어야 한다.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); GUARD=$(readlink -f "${1:-$HERE/../colab-guard}")
G=$(mktemp -d); trap 'rm -rf "$G"' EXIT
mkdir -p "$G/home/.local/bin" "$G/home/fake" "$G/tools"
cp "$HERE/fakecolab" "$G/home/.local/bin/colab"; chmod +x "$G/home/.local/bin/colab"
printf '#!/bin/sh\nexit 0\n' > "$G/home/.local/bin/notify-send"; chmod +x "$G/home/.local/bin/notify-send"   # 시험 중 실제 알림을 띄우지 않는다
printf '#!/usr/bin/env bash\ncolab exec -s "$1" -f /dev/null --timeout 5; echo "saved $1"\n' > "$G/tools/saver.sh"
cp "$HERE/cases.sh" "$G/tools/t.sh"
bash "$G/tools/t.sh" "$GUARD"
