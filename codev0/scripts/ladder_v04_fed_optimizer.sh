#!/usr/bin/env bash
# Federated-optimizer ladder for V0.4 on Computers and Amazon-ratings.
#
# Question this answers: the O4 block-v protocol helps on Cora (81.4660 federated
# vs 79.5332 local) but destroys Computers (79.7384 federated vs 89.3329 local).
# Is that caused by the block-v second-moment federation specifically, or by
# federating at all? The two anchors are already measured, so this fills in the
# three plain protocols between them:
#
#   no federation (--local-only)      89.3329 / 42.5505   (measured)
#   persistent_adam                   ?                    (this script)
#   reset_adam                        ?                    (this script)
#   reset_adamw                       ?                    (this script)
#   block_v (O4, default)             79.7384 / 40.9068   (measured)
#
# All rows keep global model replacement active; only the optimizer's state
# handling differs, so any movement is attributable to the protocol. Same
# reasoning as before: full 100 rounds, because a 50-round screen is not a valid
# proxy here.

set -euo pipefail

BASE_PATH="${BASE_PATH:-/opt/data/private/xzc/work2}"
PYTHON_BIN="${PYTHON_BIN:-/root/anaconda3/envs/torch/bin/python}"
CAMPAIGN="${CAMPAIGN:-v04_fed_optimizer_ladder}"
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
modes=(reset_adamw reset_adam persistent_adam)

if [[ ! -f "${STATUS_FILE}" ]]; then
  printf 'timestamp\tdataset\tmode\tstatus\tartifact\tacc_mean_pct\tacc_std_pct\tmacro_f1_mean_pct\n' >"${STATUS_FILE}"
fi

record() {
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -u +%FT%TZ)" "$1" "$2" "$3" "$4" "$5" "$6" "$7" >>"${STATUS_FILE}"
}

run_one() {
  local dataset="$1" mode="$2"
  local key="${dataset}__${mode}"
  local marker="${STATUS_DIR}/${key}.done"
  local run_log="${RUN_LOG_DIR}/${key}.log"

  if [[ -f "${marker}" ]]; then
    printf '[ladder] skip completed %s\n' "${key}"
    return 0
  fi

  printf '[ladder] start dataset=%s mode=%s rounds=%s time=%s\n' \
    "${dataset}" "${mode}" "${ROUNDS}" "$(date -u +%FT%TZ)"

  if PYTHONUNBUFFERED=1 "${PYTHON_BIN}" "${BASE_PATH}/codev0/main.py" \
      --model v04 --dataset "${dataset}" \
      --n-clients 10 --n-workers 10 --n-rnds "${ROUNDS}" \
      --frac 1.0 --gpu 0,1 --aggregation equal --seed 42 \
      --fed-optimizer "${mode}" \
      --run-tag "ladder_${mode}_${ROUNDS}r" >"${run_log}" 2>&1; then
    local artifact
    artifact="$({
      find "${BASE_PATH}/logs/${dataset}_disjoint/clients_10" \
        -maxdepth 1 -type d -name "*_v04_ladder_${mode}_${ROUNDS}r" -print
    } | sort | tail -n 1)"
    if [[ -n "${artifact}" && -f "${artifact}/result.json" ]]; then
      read -r acc std f1 < <("${PYTHON_BIN}" -c "
import json
r=json.load(open('${artifact}/result.json'))
print(r['mean']*100, r['std']*100, r['f1_mean']*100)
")
      printf '%s\n' "${artifact}" >"${marker}"
      record "${dataset}" "${mode}" completed "${artifact}" "${acc}" "${std}" "${f1}"
      printf '[ladder] done %s ACC=%.4f F1=%.4f\n' "${key}" "${acc}" "${f1}"
    else
      record "${dataset}" "${mode}" missing_result "${run_log}" "" "" ""
      printf '[ladder] missing result %s\n' "${key}"
    fi
  else
    local code=$?
    record "${dataset}" "${mode}" "failed_${code}" "${run_log}" "" "" ""
    printf '[ladder] failed %s exit=%s\n' "${key}" "${code}"
  fi
}

printf '[ladder] campaign=%s rounds=%s started=%s\n' \
  "${CAMPAIGN}" "${ROUNDS}" "$(date -u +%FT%TZ)"

for dataset in "${datasets[@]}"; do
  for mode in "${modes[@]}"; do
    run_one "${dataset}" "${mode}"
  done
done

"${PYTHON_BIN}" - "$STATUS_FILE" "$SUMMARY_FILE" "$BASE_PATH" <<'PY'
import csv, os, sys
status, out, base = sys.argv[1], sys.argv[2], sys.argv[3]
rows = [r for r in csv.DictReader(open(status), delimiter='\t')
        if r['status'] == 'completed' and r['acc_mean_pct']]
measured = {}
for r in rows:
    measured.setdefault(r['dataset'], {})[r['mode']] = r

# Anchors measured elsewhere: the all11 baseline is block_v federation, and the
# protocol isolation campaign holds the no-federation ceiling.
anchors = {}
for path, label, field in [
    (os.path.join(base, 'logs/batch_runs', '20260911_all11_reference_o4', 'summary.csv'), 'block_v', 'model'),
    (os.path.join(base, 'logs/batch_runs', 'v04_protocol_isolation', 'summary.csv'), 'local_only', 'config'),
]:
    if not os.path.isfile(path):
        continue
    for r in csv.DictReader(open(path)):
        if field == 'model' and r.get('model') != 'v04':
            continue
        if field == 'config' and r.get('federation') != 'off':
            continue
        anchors.setdefault(r['dataset'], {})[label] = r

order = ['local_only', 'persistent_adam', 'reset_adam', 'reset_adamw', 'block_v']
with open(out, 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['dataset', 'protocol', 'federation', 'acc_mean_pct',
                'acc_std_pct', 'macro_f1_mean_pct', 'source'])
    for dataset in ('Computers', 'Amazon-ratings'):
        for label in order:
            r = measured.get(dataset, {}).get(label)
            if r:
                w.writerow([dataset, label, 'on', r['acc_mean_pct'], r['acc_std_pct'],
                            r['macro_f1_mean_pct'], 'this_campaign'])
                continue
            a = anchors.get(dataset, {}).get(label)
            if a:
                w.writerow([dataset, label, 'off' if label == 'local_only' else 'on',
                            a['acc_mean_pct'], a['acc_std_pct'],
                            a['macro_f1_mean_pct'], 'anchor'])
print(f'[ladder] summary written to {out}')
PY

printf '[ladder] finished=%s\n' "$(date -u +%FT%TZ)"
