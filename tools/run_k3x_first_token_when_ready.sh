#!/usr/bin/env bash
# 96개 fragment 봉인 후 K3X mixed-precision 첫 토큰을 자동 생성합니다.
set -euo pipefail

cd /mnt/c/Users/jolib/Documents/project-k3x/.worktrees/milestone-twenty-four-cuda-graph-cache
mkdir -p /mnt/d/K3X-k3x-first-token/{objects,state,logs} \
  results/b0045-k3x-first-token-progress results/b0046-k3x-first-token
exec >>/mnt/d/K3X-k3x-first-token/logs/stdout.log \
  2>>/mnt/d/K3X-k3x-first-token/logs/stderr.log

export PYTHONPATH=.:converter:reference
python=/home/jolib/.venvs/k3x-m1/bin/python
ledger=/mnt/c/K3X/immortal-ledger-quality.json
set_path=/mnt/c/K3X/shards/model.k3xset
topology=artifacts/m37-local-foundry/official-topology.json
objects=/mnt/d/K3X-k3x-first-token/objects
state=/mnt/d/K3X-k3x-first-token/state

while true; do
  completed=$("$python" -c \
    'import json,sys; print(len(json.load(open(sys.argv[1]))["completed_units"]))' \
    "$ledger")
  if [[ "$completed" == "96" ]]; then
    break
  fi
  sleep 30
done

"$python" tools/write_fragment_set.py \
  --manifest artifacts/m37-local-foundry/source-manifest.json \
  --destination /mnt/c/K3X/shards \
  --ledger "$ledger" \
  --output "$set_path" \
  --output-budget-bytes 1510500000000

if [[ ! -f "$state/state.json" ]]; then
  "$python" tools/run_official_layer0.py \
    --topology "$topology" \
    --object-dir "$objects" \
    --state-dir "$state" \
    --output results/b0045-k3x-first-token-progress/layer-00.json \
    --token-id 1 \
    --k3x-set "$set_path"
fi

"$python" tools/run_official_remaining.py \
  --topology "$topology" \
  --object-dir "$objects" \
  --state-dir "$state" \
  --result-dir results/b0045-k3x-first-token-progress \
  --k3x-set "$set_path"

"$python" tools/run_official_head.py \
  --topology "$topology" \
  --object-dir "$objects" \
  --state-dir "$state" \
  --output results/b0046-k3x-first-token/summary.json \
  --k3x-set "$set_path"
