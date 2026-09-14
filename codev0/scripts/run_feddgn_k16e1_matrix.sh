#!/usr/bin/env bash
# Official-parameter A-DGN matrix with the user-requested 100 x 1 budget.

set -euo pipefail

BASE_PATH="${BASE_PATH:-/opt/data/private/xzc/work2}"
GPU="${GPU:-0,1}"
N_WORKERS="${N_WORKERS:-10}"
ROUNDS="${ROUNDS:-100}"
CAMPAIGN_PREFIX="${CAMPAIGN_PREFIX:-20260914_feddgn_k16e1}"
RUNNER="${BASE_PATH}/codev0/scripts/run_feddgn_core100.sh"
CONTROL_DIR="${BASE_PATH}/logs/batch_runs/${CAMPAIGN_PREFIX}_matrix"

if (( $# > 0 )); then
  datasets=("$@")
else
  datasets=(Cora CiteSeer Roman-empire)
fi

mkdir -p "${CONTROL_DIR}"
printf 'started=%s\n' "$(date -u +%FT%TZ)" >"${CONTROL_DIR}/started.txt"

run_protocol() {
  local model_variant="$1"
  local optimizer_mode="$2"
  local label="$3"
  local campaign="${CAMPAIGN_PREFIX}_${label}_core${ROUNDS}_w${N_WORKERS}"

  printf '[feddgn-k16e1] start label=%s campaign=%s time=%s\n' \
    "${label}" "${campaign}" "$(date -u +%FT%TZ)"
  BASE_PATH="${BASE_PATH}" \
  MODEL_VARIANT="${model_variant}" \
  FEDDGN_FED_OPTIMIZER="${optimizer_mode}" \
  CAMPAIGN="${campaign}" \
  ROUNDS="${ROUNDS}" \
  GPU="${GPU}" \
  N_WORKERS="${N_WORKERS}" \
    bash "${RUNNER}" "${datasets[@]}"
}

run_protocol feddgn_local persistent_adam local
run_protocol feddgn_avg persistent_adam avg_persistent_adam
run_protocol feddgn_avg reset_adam avg_reset_adam
run_protocol feddgn_avg block_v avg_block_v

printf 'completed=%s\n' "$(date -u +%FT%TZ)" >"${CONTROL_DIR}/done.txt"
printf '[feddgn-k16e1] finished=%s\n' "$(date -u +%FT%TZ)"
