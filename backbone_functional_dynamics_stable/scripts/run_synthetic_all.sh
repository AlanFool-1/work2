#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/opt/data/private/xzc/work2/backbone}"
BASE_PATH="${BASE_PATH:-/opt/data/private/xzc/work2}"
PYTHON="${PYTHON:-/root/anaconda3/envs/torch/bin/python}"
GPU="${GPU:-0,1}"

scenarios=(
  iid
  label_shift
  feature_shift
  feature_mixed
  structure_homophily
  structure_degree
  structure_mixed
  mixed
)

for scenario in "${scenarios[@]}"; do
  echo "[synthetic] running ${scenario}"
  "${PYTHON}" "${PROJECT_ROOT}/main.py" \
    --base-path "${BASE_PATH}" \
    --dataset Synthetic \
    --synthetic-scenario "${scenario}" \
    --gpu "${GPU}" \
    --n-clients 10 \
    --n-workers 10 \
    --run-tag "synthetic_${scenario}"
done
