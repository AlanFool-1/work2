#!/usr/bin/env bash
# V0.4 protocol isolation on the two datasets where it fails to learn.
#
# Motivation: on Computers, V0.4 sits at 67% test ACC for all 100 rounds while
# the ordinary-GCN `local` entry on the same partitions climbs to 86.5%. All
# entries start from the same round-10 accuracy, so the data and the encoder
# path are fine; V0.4's own protocol is what blocks progress. Hyperparameter
# tuning did not move either dataset (see the v04_tune_computers_amazon
# campaign), so this run isolates structure instead.
#
# There is one variable here, federation on or off:
#
#   federation on   existing all11 campaign baseline   (already measured)
#   federation off  local_only                         (this script)
#
# The Lie / reference-field ablations are deliberately NOT run. Runs are the
# full 100 rounds: a 50-round screen is not a valid proxy in this codebase,
# because the baseline gains +8.35pp between round 50 and 100 while the
# high-lr configs gain only +3.28pp, so short screens systematically prefer
# configurations that plateau early.

set -euo pipefail

BASE_PATH="${BASE_PATH:-/opt/data/private/xzc/work2}"
PYTHON_BIN="${PYTHON_BIN:-/root/anaconda3/envs/torch/bin/python}"
CAMPAIGN="${CAMPAIGN:-v04_protocol_isolation}"
CONTROL_DIR="${BASE_PATH}/logs/batch_runs/${CAMPAIGN}"
STATUS_DIR="${CONTROL_DIR}/status"
RUN_LOG_DIR="${CONTROL_DIR}/runs"
DRIVER_LOG="${CONTROL_DIR}/driver.log"
STATUS_FILE="${CONTROL_DIR}/status.tsv"
SUMMARY_FILE="${CONTROL_DIR}/summary.csv"
ROUNDS="${ROUNDS:-100}"

mkdir -p "${STATUS_DIR}" "${RUN_LOG_DIR}"
exec >>"${DRIVER_LOG}" 2>&1

datasets=(Computers Amazon-ratings)

# name|extra CLI args
configs=(
  "local_only|--local-only"
)

if [[ ! -f "${STATUS_FILE}" ]]; then
  printf 'timestamp\tdataset\tconfig\tstatus\tartifact\tacc_mean_pct\tacc_std_pct\tmacro_f1_mean_pct\n' >"${STATUS_FILE}"
fi

record() {
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -u +%FT%TZ)" "$1" "$2" "$3" "$4" "$5" "$6" "$7" >>"${STATUS_FILE}"
}

run_one() {
  local dataset="$1" config="$2" args="$3"
  local key="${dataset}__${config}"
  local marker="${STATUS_DIR}/${key}.done"
  local run_log="${RUN_LOG_DIR}/${key}.log"

  if [[ -f "${marker}" ]]; then
    printf '[iso] skip completed %s\n' "${key}"
    return 0
  fi

  printf '[iso] start dataset=%s config=%s rounds=%s time=%s\n' \
    "${dataset}" "${config}" "${ROUNDS}" "$(date -u +%FT%TZ)"

  # shellcheck disable=SC2086
  if PYTHONUNBUFFERED=1 "${PYTHON_BIN}" "${BASE_PATH}/codev0/main.py" \
      --model v04 --dataset "${dataset}" \
      --n-clients 10 --n-workers 10 --n-rnds "${ROUNDS}" \
      --frac 1.0 --gpu 0,1 --aggregation equal --seed 42 \
      --run-tag "iso_${config}_${ROUNDS}r" ${args} >"${run_log}" 2>&1; then
    local artifact
    artifact="$({
      find "${BASE_PATH}/logs/${dataset}_disjoint/clients_10" \
        -maxdepth 1 -type d -name "*_v04_iso_${config}_${ROUNDS}r" -print
    } | sort | tail -n 1)"
    if [[ -n "${artifact}" && -f "${artifact}/result.json" ]]; then
      read -r acc std f1 < <("${PYTHON_BIN}" -c "
import json
r=json.load(open('${artifact}/result.json'))
print(r['mean']*100, r['std']*100, r['f1_mean']*100)
")
      printf '%s\n' "${artifact}" >"${marker}"
      record "${dataset}" "${config}" completed "${artifact}" "${acc}" "${std}" "${f1}"
      printf '[iso] done %s ACC=%.4f F1=%.4f\n' "${key}" "${acc}" "${f1}"
    else
      record "${dataset}" "${config}" missing_result "${run_log}" "" "" ""
      printf '[iso] missing result %s\n' "${key}"
    fi
  else
    local code=$?
    record "${dataset}" "${config}" "failed_${code}" "${run_log}" "" "" ""
    printf '[iso] failed %s exit=%s\n' "${key}" "${code}"
  fi
}

printf '[iso] campaign=%s rounds=%s started=%s\n' \
  "${CAMPAIGN}" "${ROUNDS}" "$(date -u +%FT%TZ)"

for dataset in "${datasets[@]}"; do
  for entry in "${configs[@]}"; do
    run_one "${dataset}" "${entry%%|*}" "${entry#*|}"
  done
done

"${PYTHON_BIN}" - "$STATUS_FILE" "$SUMMARY_FILE" "$BASE_PATH" <<'PY'
import csv, json, os, sys
status, out, base = sys.argv[1], sys.argv[2], sys.argv[3]
rows = [r for r in csv.DictReader(open(status), delimiter='\t')
        if r['status'] == 'completed' and r['acc_mean_pct']]

# The "federation on / Lie on" cell is the existing all11 campaign baseline.
baseline = {}
for r in csv.DictReader(open(os.path.join(base, 'logs/batch_runs',
                                          '20260911_all11_reference_o4', 'summary.csv'))):
    if r['model'] == 'v04' and r.get('status') == 'completed':
        baseline[r['dataset']] = r

with open(out, 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['dataset', 'federation', 'config', 'acc_mean_pct',
                'acc_std_pct', 'macro_f1_mean_pct', 'artifact'])
    for dataset in ('Computers', 'Amazon-ratings'):
        b = baseline.get(dataset)
        if b:
            w.writerow([dataset, 'on', 'all11_baseline', b['acc_mean_pct'],
                        b['acc_std_pct'], b['macro_f1_mean_pct'], b['artifact']])
        for r in rows:
            if r['dataset'] == dataset:
                w.writerow([dataset, 'off', r['config'], r['acc_mean_pct'],
                            r['acc_std_pct'], r['macro_f1_mean_pct'], r['artifact']])
print(f'[iso] summary written to {out}')
PY

printf '[iso] finished=%s\n' "$(date -u +%FT%TZ)"
