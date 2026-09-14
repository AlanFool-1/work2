#!/usr/bin/env bash
# Persistent 100-round local FedDGN evaluation on the three target datasets.

set -euo pipefail

BASE_PATH="${BASE_PATH:-/opt/data/private/xzc/work2}"
PYTHON_BIN="${PYTHON_BIN:-/root/anaconda3/envs/torch/bin/python}"
MODEL_VARIANT="${MODEL_VARIANT:-feddgn_local}"
CAMPAIGN="${CAMPAIGN:-20260914_${MODEL_VARIANT}_core100_w10}"
ROUNDS="${ROUNDS:-100}"
GPU="${GPU:-0,1}"
N_WORKERS="${N_WORKERS:-10}"
FEDDGN_FED_OPTIMIZER="${FEDDGN_FED_OPTIMIZER:-reset_adam}"

CONTROL_DIR="${BASE_PATH}/logs/batch_runs/${CAMPAIGN}"
RUN_LOG_DIR="${CONTROL_DIR}/runs"
STATUS_DIR="${CONTROL_DIR}/status"
STATUS_FILE="${CONTROL_DIR}/status.tsv"
DONE_FILE="${CONTROL_DIR}/done.txt"
CONFIG_FILE="${BASE_PATH}/codev0/configs/${MODEL_VARIANT}.json"

case "${MODEL_VARIANT}" in
  feddgn_local)
    protocol_mode='local_only'
    model_args=(--local-only)
    ;;
  feddgn_avg)
    protocol_mode="fedavg_${FEDDGN_FED_OPTIMIZER}"
    model_args=()
    ;;
  *)
    printf '[feddgn-batch] unsupported MODEL_VARIANT=%s\n' \
      "${MODEL_VARIANT}" >&2
    exit 2
    ;;
esac

if (( $# > 0 )); then
  datasets=("$@")
else
  datasets=(Cora CiteSeer Roman-empire)
fi

mkdir -p "${RUN_LOG_DIR}" "${STATUS_DIR}"
if [[ ! -f "${STATUS_FILE}" ]]; then
  printf 'timestamp\tdataset\tworkers\tstatus\tartifact\tmetric\tmean_pct\tstd_pct\n' \
    >"${STATUS_FILE}"
fi

record_status() {
  local dataset="$1"
  local status="$2"
  local artifact="$3"
  local metric="$4"
  local mean_pct="$5"
  local std_pct="$6"
  local temporary="${STATUS_FILE}.tmp.$$"

  awk -F '\t' -v dataset="${dataset}" \
    'NR == 1 || $2 != dataset' "${STATUS_FILE}" >"${temporary}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -u +%FT%TZ)" "${dataset}" "${N_WORKERS}" "${status}" \
    "${artifact}" "${metric}" "${mean_pct}" "${std_pct}" >>"${temporary}"
  mv "${temporary}" "${STATUS_FILE}"
}

printf '[feddgn-batch] campaign=%s pid=%s started=%s\n' \
  "${CAMPAIGN}" "$$" "$(date -u +%FT%TZ)"
printf '[feddgn-batch] datasets=%s\n' "${datasets[*]}"
printf '%s\n' \
  "[feddgn-batch] model=${MODEL_VARIANT} protocol=10_clients,${ROUNDS}_rounds,1_local_epoch,seed_42,${protocol_mode}" \
  "[feddgn-batch] config=${CONFIG_FILE} gpu=${GPU} workers=${N_WORKERS} optimizer=${FEDDGN_FED_OPTIMIZER}"

for dataset in "${datasets[@]}"; do
  marker="${STATUS_DIR}/${dataset}.done"
  run_log="${RUN_LOG_DIR}/${dataset}.log"
  run_tag="${CAMPAIGN}_${dataset}"

  if [[ -f "${marker}" ]]; then
    artifact="$(<"${marker}")"
    if [[ -f "${artifact}/result.json" ]]; then
      printf '[feddgn-batch] skip completed dataset=%s artifact=%s\n' \
        "${dataset}" "${artifact}"
      continue
    fi
  fi

  record_status "${dataset}" running "${run_log}" '' '' ''
  printf '[feddgn-batch] start dataset=%s time=%s\n' \
    "${dataset}" "$(date -u +%FT%TZ)"

  command=(
    "${PYTHON_BIN}" "${BASE_PATH}/codev0/main.py"
    --model "${MODEL_VARIANT}"
    --dataset "${dataset}"
    --config "${CONFIG_FILE}"
    --n-clients 10
    --n-workers "${N_WORKERS}"
    --n-rnds "${ROUNDS}"
    --n-eps 1
    --frac 1.0
    --gpu "${GPU}"
    --aggregation equal
    --seed 42
    --log-profile minimal
    --feddgn-fed-optimizer "${FEDDGN_FED_OPTIMIZER}"
    "${model_args[@]}"
    --run-tag "${run_tag}"
  )

  if PYTHONUNBUFFERED=1 "${command[@]}" >"${run_log}" 2>&1; then
    artifact="$({
      find "${BASE_PATH}/logs/${dataset}_disjoint/clients_10" \
        -maxdepth 1 -type d -name "*_${MODEL_VARIANT}_${run_tag}" -print
    } | sort | tail -n 1)"
    if [[ -n "${artifact}" && -f "${artifact}/result.json" ]]; then
      read -r metric mean_pct std_pct < <(
        "${PYTHON_BIN}" - "${artifact}/result.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding='utf-8') as stream:
    result = json.load(stream)
print(result['metric'], result['mean'] * 100.0, result['std'] * 100.0)
PY
      )
      printf '%s\n' "${artifact}" >"${marker}"
      record_status \
        "${dataset}" completed "${artifact}" "${metric}" \
        "${mean_pct}" "${std_pct}"
      printf '[feddgn-batch] completed dataset=%s %s=%s+/-=%s artifact=%s\n' \
        "${dataset}" "${metric}" "${mean_pct}" "${std_pct}" "${artifact}"
    else
      record_status "${dataset}" missing_result "${run_log}" '' '' ''
      printf '[feddgn-batch] missing result dataset=%s log=%s\n' \
        "${dataset}" "${run_log}"
      exit 2
    fi
  else
    exit_code=$?
    record_status "${dataset}" "failed_${exit_code}" "${run_log}" '' '' ''
    printf '[feddgn-batch] failed dataset=%s exit=%s log=%s\n' \
      "${dataset}" "${exit_code}" "${run_log}"
    exit "${exit_code}"
  fi
done

printf 'completed=%s\n' "$(date -u +%FT%TZ)" >"${DONE_FILE}"
printf '[feddgn-batch] finished=%s\n' "$(date -u +%FT%TZ)"
