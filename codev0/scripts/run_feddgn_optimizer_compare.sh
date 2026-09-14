#!/usr/bin/env bash
# Run the two missing FedDGN optimizer-state protocols sequentially.
# The reset_adam 100-round baseline already exists and is intentionally reused.

set -euo pipefail

BASE_PATH="${BASE_PATH:-/opt/data/private/xzc/work2}"
GPU="${GPU:-0,1}"
N_WORKERS="${N_WORKERS:-10}"
ROUNDS="${ROUNDS:-100}"
CAMPAIGN_PREFIX="${CAMPAIGN_PREFIX:-20260914_feddgn_avg_optimizer}"
RUNNER="${BASE_PATH}/codev0/scripts/run_feddgn_core100.sh"
CONTROL_DIR="${BASE_PATH}/logs/batch_runs/${CAMPAIGN_PREFIX}_missing2"

if (( $# > 0 )); then
  datasets=("$@")
else
  datasets=(Cora CiteSeer Roman-empire)
fi

mkdir -p "${CONTROL_DIR}"
printf 'started=%s\n' "$(date -u +%FT%TZ)" >"${CONTROL_DIR}/started.txt"

for optimizer_mode in persistent_adam block_v; do
  campaign="${CAMPAIGN_PREFIX}_${optimizer_mode}_core${ROUNDS}_w${N_WORKERS}"
  printf '[feddgn-optimizer-compare] start mode=%s campaign=%s time=%s\n' \
    "${optimizer_mode}" "${campaign}" "$(date -u +%FT%TZ)"
  BASE_PATH="${BASE_PATH}" \
  MODEL_VARIANT=feddgn_avg \
  FEDDGN_FED_OPTIMIZER="${optimizer_mode}" \
  CAMPAIGN="${campaign}" \
  ROUNDS="${ROUNDS}" \
  GPU="${GPU}" \
  N_WORKERS="${N_WORKERS}" \
    bash "${RUNNER}" "${datasets[@]}"
done

printf 'completed=%s\n' "$(date -u +%FT%TZ)" >"${CONTROL_DIR}/done.txt"
printf '[feddgn-optimizer-compare] finished=%s\n' "$(date -u +%FT%TZ)"
