#!/usr/bin/env bash
# 보존된 layer 24 state에서 공식 원본 정밀도 기준 토큰 생성을 재개합니다.
set -euo pipefail

cd /mnt/c/Users/jolib/Documents/project-k3x/.worktrees/milestone-twenty-four-cuda-graph-cache
export PYTHONPATH=.:converter:reference
python=/home/jolib/.venvs/k3x-m1/bin/python
topology=artifacts/m37-local-foundry/official-topology.json
objects=/mnt/d/K3X-reference-objects
state=artifacts/m36-official-first-token/state
results=results/b0043-official-first-token-progress

mkdir -p "$objects" "$results" results/b0044-official-first-token
"$python" tools/run_official_remaining.py \
  --topology "$topology" \
  --object-dir "$objects" \
  --state-dir "$state" \
  --result-dir "$results"
"$python" tools/run_official_head.py \
  --topology "$topology" \
  --object-dir "$objects" \
  --state-dir "$state" \
  --output results/b0044-official-first-token/summary.json
