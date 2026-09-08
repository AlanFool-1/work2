#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/anaconda3/envs/torch/bin/python}"
CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_ROOT="$(cd "${CODE_ROOT}/.." && pwd)"
OUTPUT_ROOT="${1:-${CODE_ROOT}/run_logs/v01_local/Cora_r32_seed42_100ep}"

mkdir -p "${OUTPUT_ROOT}"

for first_client in 0 2 4 6 8; do
  second_client=$((first_client + 1))
  first_dir="${OUTPUT_ROOT}/client_${first_client}"
  second_dir="${OUTPUT_ROOT}/client_${second_client}"
  mkdir -p "${first_dir}" "${second_dir}"
  echo "[local-grid] clients ${first_client} and ${second_client}"

  "${PYTHON_BIN}" "${CODE_ROOT}/scripts/run_v01_single_client.py" \
    --base-path "${WORK_ROOT}" --dataset Cora \
    --client-id "${first_client}" --n-clients 10 --device cuda:0 \
    --seed 42 --epochs 100 --patience 101 --latent-dim 32 \
    --reconstruction-weight 0 --output-dir "${first_dir}" \
    >"${first_dir}/train.log" 2>&1 &
  first_pid=$!

  "${PYTHON_BIN}" "${CODE_ROOT}/scripts/run_v01_single_client.py" \
    --base-path "${WORK_ROOT}" --dataset Cora \
    --client-id "${second_client}" --n-clients 10 --device cuda:1 \
    --seed 42 --epochs 100 --patience 101 --latent-dim 32 \
    --reconstruction-weight 0 --output-dir "${second_dir}" \
    >"${second_dir}/train.log" 2>&1 &
  second_pid=$!

  wait "${first_pid}"
  wait "${second_pid}"
done

"${PYTHON_BIN}" "${CODE_ROOT}/scripts/summarize_v01_local.py" \
  "${OUTPUT_ROOT}"
echo "[local-grid] results: ${OUTPUT_ROOT}/summary.md"
