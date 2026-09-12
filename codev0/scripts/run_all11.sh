#!/usr/bin/env bash
set -euo pipefail

BASE_PATH="${BASE_PATH:-/opt/data/private/xzc/work2}"
PYTHON_BIN="${PYTHON_BIN:-/root/anaconda3/envs/torch/bin/python}"
CAMPAIGN="${CAMPAIGN:-20260911_all11_reference_persistent_personalized}"
CONTROL_DIR="${BASE_PATH}/logs/batch_runs/${CAMPAIGN}"
STATUS_DIR="${CONTROL_DIR}/status"
RUN_LOG_DIR="${CONTROL_DIR}/runs"
DRIVER_LOG="${CONTROL_DIR}/driver.log"
STATUS_FILE="${CONTROL_DIR}/status.tsv"

mkdir -p "${STATUS_DIR}" "${RUN_LOG_DIR}"
exec >>"${DRIVER_LOG}" 2>&1

models=(local fedavg fedpub fedaux v04)
datasets=(
  Cora CiteSeer PubMed Computers Photo Roman-empire Amazon-ratings
  Minesweeper Tolokers Questions ogbn-arxiv
)

if [[ ! -f "${STATUS_FILE}" ]]; then
  printf 'timestamp\tmodel\tdataset\tstatus\tartifact\n' >"${STATUS_FILE}"
fi

record_status() {
  local model="$1"
  local dataset="$2"
  local status="$3"
  local artifact="$4"
  local tmp_file="${STATUS_FILE}.tmp.$$"
  awk -F '\t' -v model="${model}" -v dataset="${dataset}" \
    'NR == 1 || $2 != model || $3 != dataset' "${STATUS_FILE}" \
    >"${tmp_file}"
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$(date -u +%FT%TZ)" "${model}" "${dataset}" "${status}" \
    "${artifact}" >>"${tmp_file}"
  mv "${tmp_file}" "${STATUS_FILE}"
}

printf '[batch] campaign=%s pid=%s started=%s\n' \
  "${CAMPAIGN}" "$$" "$(date -u +%FT%TZ)"
printf '%s\n' \
  '[batch] protocol=10_clients,100_rounds,1_local_epoch,seed_42,equal;' \
  '[batch] optimizers=local:persistent_adam,fedavg/v04:O4_block_v,fedpub/fedaux:persistent_adam'

for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    key="${model}__${dataset}"
    done_marker="${STATUS_DIR}/${key}.done"
    run_log="${RUN_LOG_DIR}/${key}.log"
    if [[ -f "${done_marker}" ]]; then
      printf '[batch] skip completed model=%s dataset=%s\n' "${model}" "${dataset}"
      continue
    fi

    run_tag="${CAMPAIGN}_${dataset}"
    n_workers=10
    # V0.4 keeps the full graph in each worker. Two workers avoid replicating
    # the ogbn-arxiv graph ten times while preserving all ten logical clients.
    if [[ "${model}" == "v04" && "${dataset}" == "ogbn-arxiv" ]]; then
      n_workers=2
    fi
    printf '[batch] start model=%s dataset=%s time=%s\n' \
      "${model}" "${dataset}" "$(date -u +%FT%TZ)"
    command=(
      "${PYTHON_BIN}" "${BASE_PATH}/codev0/main.py"
      --model "${model}"
      --dataset "${dataset}"
      --n-clients 10
      --n-workers "${n_workers}"
      --n-rnds 100
      --n-eps 1
      --frac 1.0
      --gpu 0,1
      --aggregation equal
      --seed 42
      --run-tag "${run_tag}"
    )

    if PYTHONUNBUFFERED=1 "${command[@]}" >"${run_log}" 2>&1; then
      artifact="$({
        find "${BASE_PATH}/logs/${dataset}_disjoint/clients_10" \
          -maxdepth 1 -type d -name "*_${model}_${run_tag}" -print
      } | sort | tail -n 1)"
      if [[ -n "${artifact}" && -f "${artifact}/result.json" ]]; then
        printf '%s\n' "${artifact}" >"${done_marker}"
        record_status "${model}" "${dataset}" completed "${artifact}"
        printf '[batch] completed model=%s dataset=%s artifact=%s\n' \
          "${model}" "${dataset}" "${artifact}"
      else
        record_status "${model}" "${dataset}" missing_result "${run_log}"
        printf '[batch] missing result model=%s dataset=%s log=%s\n' \
          "${model}" "${dataset}" "${run_log}"
      fi
    else
      exit_code=$?
      record_status "${model}" "${dataset}" "failed_${exit_code}" "${run_log}"
      printf '[batch] failed model=%s dataset=%s exit=%s log=%s\n' \
        "${model}" "${dataset}" "${exit_code}" "${run_log}"
    fi
  done
done

printf '[batch] finished=%s\n' "$(date -u +%FT%TZ)"
