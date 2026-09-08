#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/anaconda3/envs/torch/bin/python}"
CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_ROOT="$(cd "${CODE_ROOT}/.." && pwd)"
OUTPUT_ROOT="${1:-${CODE_ROOT}/run_logs/v01_gate0/gate0_20260908_seed42}"
EPOCHS="${EPOCHS:-200}"
PATIENCE="${PATIENCE:-50}"

mkdir -p "${OUTPUT_ROOT}"

variants=(m1_r32 a1_r16 a1_r64 a2_r32_rec10)
latent_dims=(32 16 64 32)
reconstruction_weights=(0 0 0 10)

for index in "${!variants[@]}"; do
  variant="${variants[$index]}"
  latent_dim="${latent_dims[$index]}"
  reconstruction_weight="${reconstruction_weights[$index]}"
  echo "[gate0-grid] starting ${variant}: Cora on cuda:0, CiteSeer on cuda:1"

  cora_dir="${OUTPUT_ROOT}/Cora/${variant}"
  citeseer_dir="${OUTPUT_ROOT}/CiteSeer/${variant}"
  mkdir -p "${cora_dir}" "${citeseer_dir}"

  "${PYTHON_BIN}" "${CODE_ROOT}/scripts/run_v01_single_client.py" \
    --base-path "${WORK_ROOT}" --dataset Cora --client-id 0 \
    --device cuda:0 --seed 42 --epochs "${EPOCHS}" --patience "${PATIENCE}" \
    --latent-dim "${latent_dim}" \
    --reconstruction-weight "${reconstruction_weight}" \
    --output-dir "${cora_dir}" >"${cora_dir}/train.log" 2>&1 &
  cora_pid=$!

  "${PYTHON_BIN}" "${CODE_ROOT}/scripts/run_v01_single_client.py" \
    --base-path "${WORK_ROOT}" --dataset CiteSeer --client-id 0 \
    --device cuda:1 --seed 42 --epochs "${EPOCHS}" --patience "${PATIENCE}" \
    --latent-dim "${latent_dim}" \
    --reconstruction-weight "${reconstruction_weight}" \
    --output-dir "${citeseer_dir}" >"${citeseer_dir}/train.log" 2>&1 &
  citeseer_pid=$!

  wait "${cora_pid}"
  wait "${citeseer_pid}"
  echo "[gate0-grid] completed ${variant}"
done

"${PYTHON_BIN}" "${CODE_ROOT}/scripts/summarize_v01_gate0.py" \
  "${OUTPUT_ROOT}"
echo "[gate0-grid] results: ${OUTPUT_ROOT}/summary.md"
