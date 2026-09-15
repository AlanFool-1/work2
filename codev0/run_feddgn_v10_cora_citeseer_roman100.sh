#!/usr/bin/env bash
set -u -o pipefail

project_root=/opt/data/private/xzc/work2
code_root=/opt/data/private/xzc/work2/codev0
campaign_root=/opt/data/private/xzc/work2/run_logs/feddgn_v10_three100_20260915_025212
config_path=/opt/data/private/xzc/work2/codev0/configs/feddgn.json
campaign_tag=feddgn_v10_o4_dictionary_fedavg_three100_20260915_025212

source /root/anaconda3/etc/profile.d/conda.sh
conda activate torch
python_bin="$(command -v python)"

mkdir -p "$campaign_root"
status_path="$campaign_root/status.tsv"
printf 'dataset\tstate\texit_code\tresult_dir\n' > "$status_path"
printf '%s\n' "$python_bin" > "$campaign_root/python.txt"
printf '%s\n' "$config_path" > "$campaign_root/config.txt"

datasets=(Cora CiteSeer Roman-empire)
success_count=0

for dataset in "${datasets[@]}"; do
    slug="${dataset//-/_}"
    dataset_log="$campaign_root/${slug}.log"
    printf '[batch] starting dataset=%s utc=%s\n' \
        "$dataset" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '%s\trunning\t\t\n' "$dataset" >> "$status_path"

    cd "$code_root" || exit 2
    PYTHONUNBUFFERED=1 "$python_bin" main.py \
        --model feddgn \
        --dataset "$dataset" \
        --config "$config_path" \
        --n-clients 10 \
        --n-workers 10 \
        --n-rnds 100 \
        --n-eps 1 \
        --gpu 0,1 \
        --log-profile diagnostic \
        --run-tag "${campaign_tag}_${slug}" \
        > "$dataset_log" 2>&1
    exit_code=$?

    result_dir="$(
        sed -n 's/^\[result\] log_dir=//p' "$dataset_log" \
            | tail -n 1 \
            | tr -d '\r'
    )"
    if [[ "$exit_code" -eq 0 && -n "$result_dir" \
          && -f "$result_dir/result.json" ]]; then
        printf '%s\tsucceeded\t0\t%s\n' \
            "$dataset" "$result_dir" >> "$status_path"
        printf 'dataset=%s\nresult_dir=%s\nutc=%s\n' \
            "$dataset" "$result_dir" \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
            > "$campaign_root/${slug}.done"
        success_count=$((success_count + 1))
        printf '[batch] completed dataset=%s result_dir=%s\n' \
            "$dataset" "$result_dir"
    else
        printf '%s\tfailed\t%s\t%s\n' \
            "$dataset" "$exit_code" "$result_dir" >> "$status_path"
        printf 'dataset=%s\nexit_code=%s\nresult_dir=%s\nutc=%s\n' \
            "$dataset" "$exit_code" "$result_dir" \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
            > "$campaign_root/${slug}.failed"
        printf '[batch] failed dataset=%s exit_code=%s log=%s\n' \
            "$dataset" "$exit_code" "$dataset_log"
    fi
done

printf 'completed=%s\ntotal=3\nutc=%s\n' \
    "$success_count" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "$campaign_root/done.txt"

if [[ "$success_count" -eq 3 ]]; then
    printf '%s\n' 'all datasets succeeded' > "$campaign_root/campaign.done"
    exit 0
fi

printf '%s\n' 'one or more datasets failed' > "$campaign_root/campaign.failed"
exit 1
