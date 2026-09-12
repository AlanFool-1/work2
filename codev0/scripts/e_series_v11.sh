#!/usr/bin/env bash
# Method 1.1 first-pass comparison series on Cora, following the design document's
# section 11. These are the four decisive points, not a sweep:
#
#   e1_default      --model v11 with configs/v11.json as shipped
#   e2_classifier_z classifier reads Z_T instead of H_T
#   e3_per_band     one PH core per spectral band instead of one shared core
#   e4_full_rank    full-rank PH core (capacity upper bound, no low-rank limit)
#
# What each one decides:
#   E1  does physical-only readout + gauge fix + shared core recover test
#       performance? Go threshold is ACC >= 78%. A flat ~73% means the problem is
#       not plain over-regularization and the architecture itself needs work.
#   E2  if `z` has clearly higher train accuracy and worse test accuracy, the
#       auxiliary-shortcut hypothesis holds.
#   E3  if per-band does not improve val/test but drives train loss to zero
#       faster, the shared core is the right default.
#   E4  if full rank still trails the local GCN / V0.4 baseline by >2pp, stop
#       discussing rank and inspect the Koopman-as-forward design instead.
#
# Every row is a full 100 rounds. The reported number is the best-validation
# paired test accuracy, which is what result.json already contains; the final
# round is deliberately not used for local capacity claims.

set -euo pipefail

BASE_PATH="${BASE_PATH:-/opt/data/private/xzc/work2}"
PYTHON_BIN="${PYTHON_BIN:-/root/anaconda3/envs/torch/bin/python}"
CAMPAIGN="${CAMPAIGN:-v11_e_series}"
CONTROL_DIR="${BASE_PATH}/logs/batch_runs/${CAMPAIGN}"
STATUS_DIR="${CONTROL_DIR}/status"
RUN_LOG_DIR="${CONTROL_DIR}/runs"
DRIVER_LOG="${CONTROL_DIR}/driver.log"
STATUS_FILE="${CONTROL_DIR}/status.tsv"
SUMMARY_FILE="${CONTROL_DIR}/summary.csv"
ROUNDS="${ROUNDS:-100}"
DATASETS="${DATASETS:-Cora}"

mkdir -p "${STATUS_DIR}" "${RUN_LOG_DIR}"
exec >>"${DRIVER_LOG}" 2>&1

# variant name -> extra CLI arguments
variants=(e1_default e2_classifier_z e3_per_band e4_full_rank)
args_for() {
  case "$1" in
    e1_default)      printf '' ;;
    e2_classifier_z) printf '%s' '--v11-classifier-on z' ;;
    e3_per_band)     printf '%s' '--v11-band-parameterization per_band' ;;
    e4_full_rank)    printf '%s' '--v11-generator-type full_rank_ph' ;;
    *) printf 'unknown variant %s\n' "$1" >&2; exit 2 ;;
  esac
}

if [[ ! -f "${STATUS_FILE}" ]]; then
  printf 'timestamp\tdataset\tvariant\tstatus\tartifact\tacc_mean_pct\tacc_std_pct\tmacro_f1_mean_pct\n' >"${STATUS_FILE}"
fi

record() {
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -u +%FT%TZ)" "$1" "$2" "$3" "$4" "$5" "$6" "$7" >>"${STATUS_FILE}"
}

run_one() {
  local dataset="$1" variant="$2"
  local key="${dataset}__${variant}"
  local marker="${STATUS_DIR}/${key}.done"
  local run_log="${RUN_LOG_DIR}/${key}.log"

  if [[ -f "${marker}" ]]; then
    printf '[e-series] skip completed %s\n' "${key}"
    return 0
  fi

  printf '[e-series] start dataset=%s variant=%s rounds=%s time=%s\n' \
    "${dataset}" "${variant}" "${ROUNDS}" "$(date -u +%FT%TZ)"

  # shellcheck disable=SC2086  # args_for intentionally returns a word list
  if PYTHONUNBUFFERED=1 "${PYTHON_BIN}" "${BASE_PATH}/codev0/main.py" \
      --model v11 --dataset "${dataset}" \
      --n-clients 10 --n-workers 10 --n-rnds "${ROUNDS}" \
      --frac 1.0 --gpu 0 --aggregation equal --seed 42 \
      $(args_for "${variant}") \
      --run-tag "${variant}_${ROUNDS}r" >"${run_log}" 2>&1; then
    local artifact
    artifact="$({
      find "${BASE_PATH}/logs/${dataset}_disjoint/clients_10" \
        -maxdepth 1 -type d -name "*_v11_${variant}_${ROUNDS}r" -print
    } | sort | tail -n 1)"
    if [[ -n "${artifact}" && -f "${artifact}/result.json" ]]; then
      read -r acc std f1 < <("${PYTHON_BIN}" -c "
import json
r=json.load(open('${artifact}/result.json'))
print(r['mean']*100, r['std']*100, r['f1_mean']*100)
")
      printf '%s\n' "${artifact}" >"${marker}"
      record "${dataset}" "${variant}" completed "${artifact}" "${acc}" "${std}" "${f1}"
      printf '[e-series] done %s ACC=%.4f F1=%.4f\n' "${key}" "${acc}" "${f1}"
    else
      record "${dataset}" "${variant}" missing_result "${run_log}" "" "" ""
      printf '[e-series] missing result %s\n' "${key}"
    fi
  else
    local code=$?
    record "${dataset}" "${variant}" "failed_${code}" "${run_log}" "" "" ""
    printf '[e-series] failed %s exit=%s\n' "${key}" "${code}"
  fi
}

printf '[e-series] campaign=%s rounds=%s started=%s\n' \
  "${CAMPAIGN}" "${ROUNDS}" "$(date -u +%FT%TZ)"

for dataset in ${DATASETS}; do
  for variant in "${variants[@]}"; do
    run_one "${dataset}" "${variant}"
  done
done

# The summary answers the section 11 questions directly, so it carries the
# mechanism columns alongside the headline accuracy: train accuracy at the last
# round (memorization), the band-1 beta and R_fro (did selective dissipation
# engage), and the within-forward growth ratio (is the PH bound respected).
"${PYTHON_BIN}" - "$STATUS_FILE" "$SUMMARY_FILE" <<'PY'
import csv, os, sys
status, out = sys.argv[1], sys.argv[2]
rows = [r for r in csv.DictReader(open(status), delimiter='\t')
        if r['status'] == 'completed' and r['acc_mean_pct']]

cols = ['train_accuracy', 'manifold_loss', 'trajectory_growth_ratio',
        'trajectory_max_ratio', 'band1_beta', 'band1_gamma', 'band1_R_fro',
        'band1_J_fro', 'band1_sym_max_eig', 'test_accuracy']

with open(out, 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['dataset', 'variant', 'acc_mean_pct', 'acc_std_pct',
                'macro_f1_mean_pct'] + [f'last_{c}' for c in cols])
    for r in rows:
        artifact = r['artifact']
        last = {}
        path = os.path.join(artifact, 'metrics.csv')
        if os.path.isfile(path):
            last = list(csv.DictReader(open(path)))[-1]
        w.writerow([r['dataset'], r['variant'], r['acc_mean_pct'],
                    r['acc_std_pct'], r['macro_f1_mean_pct']] +
                   [last.get(c, '') for c in cols])
print(f'[e-series] summary written to {out}')
PY

printf '[e-series] finished=%s\n' "$(date -u +%FT%TZ)"
