#!/usr/bin/env bash
# 공식 Kimi K3의 남은 레이어를 WSL 환경에서 재개 실행합니다.
set -euo pipefail

cd /mnt/c/Users/jolib/Documents/project-k3x/.worktrees/milestone-twenty-four-cuda-graph-cache
export PYTHONPATH=.:converter:reference
exec /home/jolib/.venvs/k3x-m1/bin/python tools/run_official_remaining.py \
  --topology results/b0037-official-topology/summary.json \
  --object-dir artifacts/m36-official-first-token/objects \
  --state-dir artifacts/m36-official-first-token/state \
  --result-dir results/b0043-official-first-token-progress
