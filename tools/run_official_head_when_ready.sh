#!/usr/bin/env bash
# 전체 93레이어 state가 준비된 경우에만 공식 LM-head token 생성을 시작합니다.
set -euo pipefail

cd /mnt/c/Users/jolib/Documents/project-k3x/.worktrees/milestone-twenty-four-cuda-graph-cache
state=artifacts/m36-official-first-token/state/state.json
while true; do
  completed=$(/home/jolib/.venvs/k3x-m1/bin/python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["completed_layer"])' "$state")
  if [[ "$completed" == "92" ]]; then
    break
  fi
  if ! pgrep -f '[r]un_official_remaining.py' >/dev/null; then
    echo "remaining execution stopped at layer $completed" >&2
    exit 1
  fi
  sleep 10
done

export PYTHONPATH=.:converter:reference
exec /home/jolib/.venvs/k3x-m1/bin/python tools/run_official_head.py \
  --topology results/b0037-official-topology/summary.json \
  --object-dir artifacts/m36-official-first-token/objects \
  --state-dir artifacts/m36-official-first-token/state \
  --output results/b0044-official-first-token/summary.json
