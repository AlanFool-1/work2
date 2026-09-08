#!/usr/bin/env bash
set -uo pipefail

project_root="/opt/data/private/xzc/work2/codev0"
python_bin="/root/anaconda3/envs/torch/bin/python"
stamp="$(date -u +%Y%m%d_%H%M%S)"
run_tag="official_adgn_k16_all11_${stamp}"
launcher_dir="${project_root}/run_logs/${run_tag}"
mkdir -p "${launcher_dir}"

datasets=(
  Cora
  CiteSeer
  PubMed
  Computers
  Photo
  ogbn-arxiv
  Roman-empire
  Amazon-ratings
  Minesweeper
  Tolokers
  Questions
)

if [[ $# -gt 0 ]]; then
  datasets=("$@")
fi

status_file="${launcher_dir}/status.tsv"
printf 'dataset\tstatus\texit_code\toutput\n' > "${status_file}"

for dataset in "${datasets[@]}"; do
  output_file="${launcher_dir}/${dataset}.out"
  printf '[start] %s %s\n' "$(date -u +%FT%TZ)" "${dataset}" \
    | tee -a "${output_file}"
  PYTHONUNBUFFERED=1 "${python_bin}" "${project_root}/main.py" \
    --dataset "${dataset}" \
    --gpu 0,1 \
    --n-workers 10 \
    --n-eps 2 \
    --ode-steps 16 \
    --adgn-step-size 0.1 \
    --ode-gamma 0.1 \
    --ode-activation tanh \
    --lr 0.015 \
    --weight-decay 0.0001 \
    --run-tag "${run_tag}_${dataset}" \
    >> "${output_file}" 2>&1
  exit_code=$?
  if [[ ${exit_code} -eq 0 ]]; then
    status="completed"
  else
    status="failed"
  fi
  printf '%s\t%s\t%s\t%s\n' \
    "${dataset}" "${status}" "${exit_code}" "${output_file}" \
    >> "${status_file}"
  printf '[end] %s %s status=%s exit=%s\n' \
    "$(date -u +%FT%TZ)" "${dataset}" "${status}" "${exit_code}" \
    | tee -a "${output_file}"
done

printf '[done] %s\n' "$(date -u +%FT%TZ)" > "${launcher_dir}/done.txt"
