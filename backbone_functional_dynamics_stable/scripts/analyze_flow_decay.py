#!/usr/bin/env python3
"""Summarize early-to-late flow proxies from completed all-11 runs."""

import argparse
import csv
import glob
import json
import math
import os
import statistics
from pathlib import Path


METRICS = (
    "delta_raw_norm",
    "injection_native_ratio",
    "delta_safe_norm",
    "cluster_distance",
)


def finite_values(rows, key):
    values = []
    for row in rows:
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def dataset_name(path):
    for part in path.parts:
        if part.endswith("_disjoint"):
            return part[: -len("_disjoint")]
    raise ValueError("cannot infer dataset from {}".format(path))


def summarize_file(path, fraction):
    by_round = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            round_id = int(float(row["round"]))
            by_round.setdefault(round_id, []).append(row)

    active_rounds = []
    for round_id in sorted(by_round):
        injection = finite_values(by_round[round_id], "injection_native_ratio")
        if any(value > 0 for value in injection):
            active_rounds.append(round_id)
    if not active_rounds:
        raise ValueError("no active flow rounds in {}".format(path))

    width = max(1, int(math.ceil(fraction * len(active_rounds))))
    early_rounds = active_rounds[:width]
    late_rounds = active_rounds[-width:]
    result = {
        "dataset": dataset_name(path),
        "path": str(path),
        "active_round_start": active_rounds[0],
        "active_round_end": active_rounds[-1],
        "window_rounds": width,
    }
    for metric in METRICS:
        early = finite_values(
            [row for round_id in early_rounds for row in by_round[round_id]], metric
        )
        late = finite_values(
            [row for round_id in late_rounds for row in by_round[round_id]], metric
        )
        if not early or not late:
            raise ValueError("missing {} values in {}".format(metric, path))
        early_median = statistics.median(early)
        late_median = statistics.median(late)
        result[metric] = {
            "early_median": early_median,
            "late_median": late_median,
            "late_over_early": late_median / early_median,
        }
    return result


def aggregate(per_dataset):
    output = {}
    for metric in METRICS:
        early = [row[metric]["early_median"] for row in per_dataset]
        late = [row[metric]["late_median"] for row in per_dataset]
        ratios = [row[metric]["late_over_early"] for row in per_dataset]
        output[metric] = {
            "declining_datasets": sum(value < 1.0 for value in ratios),
            "dataset_count": len(ratios),
            "median_early": statistics.median(early),
            "median_late": statistics.median(late),
            "median_late_over_early": statistics.median(ratios),
        }
    return output


def main():
    default_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", type=Path, default=default_root)
    parser.add_argument("--window-fraction", type=float, default=0.2)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if not 0 < args.window_fraction <= 0.5:
        raise ValueError("--window-fraction must be in (0, 0.5]")

    pattern = str(
        args.workspace_root
        / "logs"
        / "*_disjoint"
        / "clients_10"
        / "*functional_dynamics_multi_proto*"
        / "functional_dynamics.csv"
    )
    paths = []
    for value in glob.glob(pattern):
        path = Path(value)
        if os.path.exists(str(path.parent / "result.json")):
            paths.append(path)
    rows = sorted(
        (summarize_file(path, args.window_fraction) for path in paths),
        key=lambda row: row["dataset"],
    )
    result = {"per_dataset": rows, "aggregate": aggregate(rows)}
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    print("completed datasets: {}".format(len(rows)))
    print("metric,declining,total,median_early,median_late,median_ratio")
    for metric in METRICS:
        item = result["aggregate"][metric]
        print(
            "{},{},{},{:.9g},{:.9g},{:.9g}".format(
                metric,
                item["declining_datasets"],
                item["dataset_count"],
                item["median_early"],
                item["median_late"],
                item["median_late_over_early"],
            )
        )
    print("dataset,injection_ratio,raw_ratio,safe_ratio,cluster_ratio")
    for row in rows:
        print(
            "{},{:.9g},{:.9g},{:.9g},{:.9g}".format(
                row["dataset"],
                row["injection_native_ratio"]["late_over_early"],
                row["delta_raw_norm"]["late_over_early"],
                row["delta_safe_norm"]["late_over_early"],
                row["cluster_distance"]["late_over_early"],
            )
        )


if __name__ == "__main__":
    main()
