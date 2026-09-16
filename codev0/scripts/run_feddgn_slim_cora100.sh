#!/usr/bin/env bash
# Cora 100-round run of the slim FedDGN (local backbone + private K fit only).
set -u -o pipefail

project_root=/opt/data/private/xzc/work2
code_root="$project_root/codev0"
config_path="$code_root/configs/feddgn.json"
campaign_root="${1:?Usage: bash run_feddgn_slim_cora100.sh CAMPAIGN_ROOT}"
case "$campaign_root" in
    "$project_root"/run_logs/feddgn_slim_cora100_*) ;;
    *) printf 'Unsafe campaign root: %s\n' "$campaign_root" >&2; exit 2 ;;
esac
if [[ -e "$campaign_root/status.tsv" ]]; then
    printf 'Campaign status already exists: %s/status.tsv\n' "$campaign_root" >&2
    exit 2
fi

source /root/anaconda3/etc/profile.d/conda.sh
conda activate torch
python_bin="$(command -v python)"
mkdir -p "$campaign_root"
printf '%s\n' "$python_bin" > "$campaign_root/python.txt"
printf '%s\n' \
    'Cora, 10 clients, 10 workers, 100 rounds x 1 epoch, seed 42' \
    'slim feddgn: local A-DGN backbone + private ResKoop K + dictionary FedAvg' \
    'server-side mode screening and K-star synthesis removed pending redesign' \
    > "$campaign_root/design.txt"
git -C "$project_root" status --short > "$campaign_root/worktree_at_start.txt"
sha256sum \
    "$config_path" \
    "$code_root/main.py" \
    "$code_root/myparser.py" \
    "$code_root/models/feddgn/client.py" \
    "$code_root/models/feddgn/server.py" \
    "$code_root/models/feddgn/reskoop.py" \
    "$code_root/models/feddgn_local/client.py" \
    "$code_root/models/feddgn_local/model.py" \
    "$code_root/models/feddgn_local/dynamics.py" \
    "$code_root/models/feddgn_local/training.py" \
    "$code_root/modules/structured_logger.py" \
    > "$campaign_root/source_sha256.txt"

status_path="$campaign_root/status.tsv"
run_log="$campaign_root/Cora.log"
printf 'dataset\tstate\texit_code\tmetrics_rounds\tacc\tresult_dir\tutc\n' \
    > "$status_path"
printf 'Cora\trunning\t\t\t\t\t%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$status_path"

cd "$code_root" || exit 2
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
    "$python_bin" main.py \
        --model feddgn \
        --dataset Cora \
        --config "$config_path" \
        --n-clients 10 \
        --n-workers 10 \
        --n-rnds 100 \
        --n-eps 1 \
        --seed 42 \
        --gpu 0 \
        --log-profile diagnostic \
        --run-tag "feddgn_slim_cora100_$(basename "$campaign_root")" \
        > "$run_log" 2>&1
exit_code=$?

result_dir="$(
    sed -n 's/^\[result\] log_dir=//p' "$run_log" | tail -n 1 | tr -d '\r'
)"
acc="$(
    sed -n 's/^\[result\] ACC=\([0-9.]*\).*/\1/p' "$run_log" | tail -n 1 | tr -d '\r'
)"
metrics_rounds=0
if [[ -n "$result_dir" && -f "$result_dir/metrics.csv" ]]; then
    metrics_rounds="$(awk 'NR > 1 { n++ } END { print n+0 }' \
        "$result_dir/metrics.csv")"
fi

if [[ "$exit_code" -eq 0 && "$metrics_rounds" -eq 100 && -n "$result_dir" ]]; then
    printf 'Cora\tsucceeded\t0\t100\t%s\t%s\t%s\n' \
        "$acc" "$result_dir" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$status_path"
    printf 'result_dir=%s\nacc=%s\nmetrics_rounds=100\nutc=%s\n' \
        "$result_dir" "$acc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        > "$campaign_root/Cora.done"
    printf '[batch] completed slim cora100 acc=%s result_dir=%s\n' \
        "$acc" "$result_dir"
    exit 0
fi

printf 'Cora\tfailed\t%s\t%s\t%s\t%s\t%s\n' \
    "$exit_code" "$metrics_rounds" "$acc" "$result_dir" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$status_path"
printf 'exit_code=%s\nmetrics_rounds=%s\nacc=%s\nresult_dir=%s\nutc=%s\n' \
    "$exit_code" "$metrics_rounds" "$acc" "$result_dir" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$campaign_root/Cora.failed"
printf '[batch] failed slim cora100 exit_code=%s log=%s\n' \
    "$exit_code" "$run_log"
exit 1
