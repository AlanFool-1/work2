#!/usr/bin/env bash
set -uo pipefail

project_root="/opt/data/private/xzc/work2/backbone_functional_dynamics_stable"
source /root/anaconda3/etc/profile.d/conda.sh
conda activate torch
python_bin="$(command -v python)"
stamp="$(date -u +%Y%m%d_%H%M%S)"
matrix_tag="functional_dynamics_multi_proto_all11_${stamp}"
matrix_dir="${project_root}/run_logs/${matrix_tag}"
status_file="${matrix_dir}/status.tsv"

datasets=(
  Cora CiteSeer PubMed Computers Photo ogbn-arxiv
  Roman-empire Amazon-ratings Minesweeper Tolokers Questions
)
if [[ $# -gt 0 ]]; then
  datasets=("$@")
fi

mkdir -p "${matrix_dir}"
printf 'dataset\tstatus\texit_code\tmetric\tpaired_test_mean\tpaired_test_std\tf1_mean\tlog_dir\n' > "${status_file}"
printf '[matrix] tag=%s\n[matrix] dir=%s\n' "${matrix_tag}" "${matrix_dir}"

for dataset in "${datasets[@]}"; do
  workers=10
  if [[ "${dataset}" == "ogbn-arxiv" ]]; then
    workers=2
  fi

  dataset_tag="${matrix_tag}_${dataset}"
  output_file="${matrix_dir}/${dataset}.log"
  printf '\n[matrix] starting dataset=%s workers=%s\n' "${dataset}" "${workers}" | tee -a "${output_file}"

  PYTHONUNBUFFERED=1 "${python_bin}" "${project_root}/main.py" \
    --dataset "${dataset}" \
    --gpu 0,1 \
    --n-clients 10 \
    --n-workers "${workers}" \
    --enable-functional-dynamics \
    --fd-num-prototypes 3 \
    --fd-min-cluster-size 2 \
    --fd-cluster-switch-margin 1.0 \
    --fd-rank 16 \
    --fd-probe-dim 4 \
    --fd-map-type regularized \
    --fd-map-max-condition 20 \
    --fd-map-orth-reg 0.01 \
    --fd-fm-steps 20 \
    --fd-normalize-generator \
    --log-profile minimal \
    --run-tag "${dataset_tag}" 2>&1 | tee -a "${output_file}"
  exit_code=${PIPESTATUS[0]}

  log_dir="$(sed -n 's/^\[result\] log_dir=//p' "${output_file}" | tail -n 1)"
  if [[ ${exit_code} -eq 0 && -n "${log_dir}" && -f "${log_dir}/result.json" ]]; then
    readarray -t fields < <("${python_bin}" - "${log_dir}/result.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    result = json.load(handle)
print(result.get("metric", ""))
print(result.get("mean", ""))
print(result.get("std", ""))
print(result.get("f1_mean", ""))
PY
    )
    printf '%s\tcomplete\t0\t%s\t%s\t%s\t%s\t%s\n' \
      "${dataset}" "${fields[0]:-}" "${fields[1]:-}" "${fields[2]:-}" \
      "${fields[3]:-}" "${log_dir}" >> "${status_file}"
    touch "${matrix_dir}/${dataset}.done"
  else
    printf '%s\tfailed\t%s\t\t\t\t\t%s\n' \
      "${dataset}" "${exit_code}" "${log_dir}" >> "${status_file}"
    touch "${matrix_dir}/${dataset}.failed"
  fi
done

touch "${matrix_dir}/matrix.done"
printf '\n[matrix] complete status=%s\n' "${status_file}"
