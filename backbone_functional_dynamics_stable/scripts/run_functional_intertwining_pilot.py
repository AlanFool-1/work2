#!/usr/bin/env python3
"""Known-map pilot for Functional Action Flow without parameter averaging.

The experiment is intentionally independent of the production federation path.
It uses small stable linear systems so the cross-client Functional Map and the
receiver's task-optimal dynamics are known.  Map fitting uses descriptor probes;
action defects use disjoint probes and horizons.
"""

import argparse
import datetime as dt
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent


def _json_dump(path, value):
    with path.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)


def _git_state():
    def run(*args):
        return subprocess.check_output(
            ["git"] + list(args), cwd=str(WORKSPACE), universal_newlines=True
        ).strip()

    try:
        return {
            "commit": run("rev-parse", "HEAD"),
            "status_short": run("status", "--short"),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "status_short": "unavailable"}


def _orthogonal(rank, generator):
    matrix = torch.randn(rank, rank, generator=generator, dtype=torch.float64)
    q_mat, r_mat = torch.linalg.qr(matrix)
    signs = torch.sign(torch.diagonal(r_mat))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return q_mat * signs.unsqueeze(0)


def _relative_norm(value, reference):
    denominator = torch.linalg.norm(reference).clamp_min(1e-12)
    return float((torch.linalg.norm(value) / denominator).item())


def _cosine(first, second):
    denominator = torch.linalg.norm(first) * torch.linalg.norm(second)
    if float(denominator) <= 1e-14:
        return 0.0
    return float((torch.sum(first * second) / denominator).item())


def _procrustes(source_descriptor, receiver_descriptor):
    left, _, right = torch.linalg.svd(
        receiver_descriptor @ source_descriptor.T, full_matrices=False
    )
    return left @ right


def _make_probes(rank, columns, coordinates, generator, scale=None):
    probes = torch.randn(rank, columns, generator=generator, dtype=torch.float64)
    mask = torch.zeros(rank, 1, dtype=torch.float64)
    mask[list(coordinates)] = 1.0
    probes = probes * mask
    if scale is not None:
        probes = probes * scale.reshape(rank, 1)
    norms = torch.linalg.norm(probes, dim=0, keepdim=True).clamp_min(1e-12)
    return probes / norms


def _make_task_probes(base_probes, transfer_scale):
    rank = base_probes.shape[0]
    probes = base_probes.clone()
    scales = torch.ones(rank, dtype=torch.float64)
    scales[2:4] = float(transfer_scale)
    probes = probes * scales.reshape(rank, 1)
    # Keep absolute excitation strength: feature heterogeneity is represented by
    # low energy in the receiver's locally under-observed coordinates.
    return probes / math.sqrt(float(rank))


def _propagators(operator, horizons):
    return {
        float(horizon): torch.matrix_exp(float(horizon) * operator)
        for horizon in horizons
    }


def _task_pairs(operator, probes, horizons):
    propagators = _propagators(operator, horizons)
    return [
        (float(horizon), probes, propagators[float(horizon)] @ probes)
        for horizon in horizons
    ]


def _action_pairs(source_operator, functional_map, probes_by_class, horizons,
                  shuffled=False, generator=None):
    propagators = _propagators(source_operator, horizons)
    reversed_horizons = list(reversed([float(value) for value in horizons]))
    result = {}
    for class_name, probes in probes_by_class.items():
        pairs = []
        if shuffled:
            permutation = torch.randperm(probes.shape[1], generator=generator)
        else:
            permutation = torch.arange(probes.shape[1])
        for index, horizon in enumerate(horizons):
            source_horizon = reversed_horizons[index] if shuffled else float(horizon)
            target_probes = probes[:, permutation] if shuffled else probes
            pairs.append((
                float(horizon),
                functional_map @ probes,
                functional_map @ (propagators[source_horizon] @ target_probes),
            ))
        result[class_name] = pairs
    return result


def _pairs_loss(operator, pairs, propagators=None):
    horizons = sorted(set(pair[0] for pair in pairs))
    propagators = propagators or _propagators(operator, horizons)
    numerator = operator.new_tensor(0.0)
    denominator = 0
    for horizon, inputs, targets in pairs:
        residual = propagators[float(horizon)] @ inputs - targets
        numerator = numerator + torch.sum(residual * residual)
        denominator += residual.numel()
    return numerator / max(denominator, 1)


def _relative_defect(operator, pairs):
    propagators = _propagators(operator, sorted(set(pair[0] for pair in pairs)))
    numerator = operator.new_tensor(0.0)
    denominator = operator.new_tensor(0.0)
    for horizon, inputs, targets in pairs:
        residual = propagators[float(horizon)] @ inputs - targets
        numerator = numerator + torch.sum(residual * residual)
        denominator = denominator + torch.sum(targets * targets)
    return float(torch.sqrt(numerator / denominator.clamp_min(1e-12)).item())


def _task_loss(operator, pairs):
    return _pairs_loss(operator, pairs)


def _gradient(operator, loss_function):
    candidate = operator.detach().clone().requires_grad_(True)
    loss = loss_function(candidate)
    gradient, = torch.autograd.grad(loss, candidate)
    return gradient.detach(), float(loss.item())


def _gate_diagnostics(initial_operator, task_train_pairs, constraints,
                      map_reliable, defect_threshold):
    task_gradient, _ = _gradient(
        initial_operator, lambda value: _task_loss(value, task_train_pairs)
    )
    gates = {}
    diagnostics = {}
    for class_name, payload in constraints.items():
        constraint_gradient, _ = _gradient(
            initial_operator, lambda value, pairs=payload["train"]:
            _pairs_loss(value, pairs)
        )
        cosine = _cosine(task_gradient, constraint_gradient)
        gate_defect = _relative_defect(initial_operator, payload["gate"])
        heldout_defect = _relative_defect(initial_operator, payload["test"])
        accepted = bool(map_reliable and gate_defect > defect_threshold and cosine > 0.0)
        gate = max(cosine, 0.0) if accepted else 0.0
        gates[class_name] = gate
        diagnostics[class_name] = {
            "gradient_cosine": cosine,
            "gate_defect": gate_defect,
            "heldout_defect": heldout_defect,
            "accepted": accepted,
            "gate": gate,
        }
    return gates, diagnostics


def _flow_gradient(operator, constraints, gates):
    active = [name for name, gate in gates.items() if gate > 0.0]
    if not active:
        return torch.zeros_like(operator)

    def objective(value):
        horizons = sorted(set(
            pair[0] for name in active for pair in constraints[name]["train"]
        ))
        propagators = _propagators(value, horizons)
        loss = value.new_tensor(0.0)
        for name in active:
            loss = loss + float(gates[name]) * _pairs_loss(
                value, constraints[name]["train"], propagators
            )
        return loss

    gradient, _ = _gradient(operator, objective)
    return -gradient


def _optimize(initial_operator, task_train_pairs, constraints, gates,
              steps, learning_rate, flow_strength, dynamic_gates=False,
              map_reliable=True, defect_threshold=0.01):
    parameter = initial_operator.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([parameter], lr=float(learning_rate))
    all_horizons = sorted(set(
        [pair[0] for pair in task_train_pairs]
        + [
            pair[0] for name in constraints
            for pair in constraints[name]["train"]
        ]
    ))
    gate_active_counts = {name: 0 for name in constraints}
    current_gates = dict(gates)
    for _ in range(int(steps)):
        if dynamic_gates:
            current_gates, _ = _gate_diagnostics(
                parameter.detach(), task_train_pairs, constraints,
                map_reliable, defect_threshold
            )
        for name, gate in current_gates.items():
            gate_active_counts[name] += int(gate > 0.0)
        optimizer.zero_grad()
        propagators = _propagators(parameter, all_horizons)
        loss = _pairs_loss(parameter, task_train_pairs, propagators)
        for name, gate in current_gates.items():
            if gate > 0.0:
                loss = loss + float(flow_strength) * float(gate) * _pairs_loss(
                    parameter, constraints[name]["train"], propagators
                )
        loss.backward()
        optimizer.step()
    if dynamic_gates:
        current_gates, _ = _gate_diagnostics(
            parameter.detach(), task_train_pairs, constraints,
            map_reliable, defect_threshold
        )
    active_fractions = {
        name: count / float(max(int(steps), 1))
        for name, count in gate_active_counts.items()
    }
    return parameter.detach(), current_gates, active_fractions


def _mean_std_ci(values):
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
    ci95 = 1.96 * std / math.sqrt(float(len(array))) if len(array) > 1 else 0.0
    return {"mean": mean, "std": std, "ci95": ci95, "n": int(len(array))}


def _aggregate(rows, group_keys, metric_keys):
    groups = {}
    for row in rows:
        key = tuple(row[item] for item in group_keys)
        groups.setdefault(key, []).append(row)
    result = []
    for key in sorted(groups):
        output = {name: value for name, value in zip(group_keys, key)}
        for metric in metric_keys:
            output[metric] = _mean_std_ci([row[metric] for row in groups[key]])
        result.append(output)
    return result


def _regime_config(name):
    if name == "feature_only":
        return {"transfer_scale": 0.12, "task_horizons": [0.5, 1.0],
                "transfer_rates": [-0.18, -1.02]}
    if name == "structure_only":
        return {"transfer_scale": 1.0, "task_horizons": [0.2, 0.4],
                "transfer_rates": [0.05, -1.25]}
    if name == "joint":
        return {"transfer_scale": 0.12, "task_horizons": [0.2, 0.4],
                "transfer_rates": [0.05, -1.25]}
    raise ValueError("unknown regime: {}".format(name))


def _build_seed(seed, rank, descriptor_columns, probe_columns, task_columns):
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    functional_map = _orthogonal(rank, generator)

    source_fit = torch.randn(
        rank, descriptor_columns, generator=generator, dtype=torch.float64
    )
    source_test = torch.randn(
        rank, descriptor_columns, generator=generator, dtype=torch.float64
    )
    noise_fit = 0.005 * torch.randn(
        rank, descriptor_columns, generator=generator, dtype=torch.float64
    )
    noise_test = 0.005 * torch.randn(
        rank, descriptor_columns, generator=generator, dtype=torch.float64
    )
    receiver_fit = functional_map @ source_fit + noise_fit
    receiver_test = functional_map @ source_test + noise_test
    learned_map = _procrustes(source_fit, receiver_fit)
    learned_inverse = _procrustes(receiver_fit, source_fit)
    wrong_map = _orthogonal(rank, generator)
    identity = torch.eye(rank, dtype=torch.float64)

    map_payloads = {
        "ground_truth": (functional_map, functional_map.T),
        "learned": (learned_map, learned_inverse),
        "wrong": (wrong_map, wrong_map.T),
        "identity": (identity, identity),
    }
    map_rows = []
    for name, (candidate, inverse) in map_payloads.items():
        descriptor_error = _relative_norm(
            candidate @ source_test - receiver_test, receiver_test
        )
        map_rows.append({
            "seed": int(seed),
            "map": name,
            "map_relative_error": _relative_norm(candidate - functional_map, functional_map),
            "heldout_descriptor_error": descriptor_error,
            "cycle_error": _relative_norm(
                inverse @ candidate - identity, identity
            ),
        })

    coordinates = {
        "satisfied": (0, 1),
        "transferable": (2, 3),
        "harmful": (4, 5),
    }
    train_probes = {
        name: _make_probes(rank, probe_columns, dims, generator)
        for name, dims in coordinates.items()
    }
    test_probes = {
        name: _make_probes(rank, probe_columns, dims, generator)
        for name, dims in coordinates.items()
    }
    gate_probes = {
        name: _make_probes(rank, probe_columns, dims, generator)
        for name, dims in coordinates.items()
    }
    task_train_base = torch.randn(
        rank, task_columns, generator=generator, dtype=torch.float64
    )
    task_test_base = torch.randn(
        rank, task_columns, generator=generator, dtype=torch.float64
    )
    return {
        "generator": generator,
        "functional_map": functional_map,
        "maps": map_payloads,
        "map_rows": map_rows,
        "map_errors": {
            row["map"]: row["heldout_descriptor_error"] for row in map_rows
        },
        "train_probes": train_probes,
        "test_probes": test_probes,
        "gate_probes": gate_probes,
        "task_train_base": task_train_base,
        "task_test_base": task_test_base,
    }


def _run_regime(seed_payload, seed, regime, args):
    rank = int(args.rank)
    generator = seed_payload["generator"]
    c_true = seed_payload["functional_map"]
    config = _regime_config(regime)

    task_rates = torch.tensor(
        [-0.20, -0.35, -0.55, -0.70, -0.85, -1.00], dtype=torch.float64
    )
    source_rates = torch.tensor(
        [-0.20, -0.35, -0.55, -0.70, -0.05, -1.70], dtype=torch.float64
    )
    initial_rates = task_rates.clone()
    initial_rates[2:4] = torch.tensor(config["transfer_rates"], dtype=torch.float64)
    initial_rates[4:6] = torch.tensor([-0.55, -1.30], dtype=torch.float64)

    source_operator = torch.diag(source_rates)
    task_operator = c_true @ torch.diag(task_rates) @ c_true.T
    initial_operator = c_true @ torch.diag(initial_rates) @ c_true.T

    task_train_source = _make_task_probes(
        seed_payload["task_train_base"],
        config["transfer_scale"]
    )
    task_test_source = _make_task_probes(
        seed_payload["task_test_base"], 1.0
    )
    task_train_pairs = _task_pairs(
        task_operator, c_true @ task_train_source, config["task_horizons"]
    )
    task_test_pairs = _task_pairs(
        task_operator, c_true @ task_test_source, args.action_horizons
    )

    constraints_by_map = {}
    gating_rows = []
    for map_name, (candidate_map, _) in seed_payload["maps"].items():
        train = _action_pairs(
            source_operator, candidate_map, seed_payload["train_probes"],
            args.action_horizons
        )
        test = _action_pairs(
            source_operator, candidate_map, seed_payload["test_probes"],
            args.action_horizons
        )
        gate = _action_pairs(
            source_operator, candidate_map, seed_payload["gate_probes"],
            args.action_horizons
        )
        constraints = {
            name: {"train": train[name], "gate": gate[name], "test": test[name]}
            for name in train
        }
        reliable = seed_payload["map_errors"][map_name] <= args.map_error_threshold
        gates, diagnostics = _gate_diagnostics(
            initial_operator, task_train_pairs, constraints,
            reliable, args.defect_threshold
        )
        constraints_by_map[map_name] = (constraints, gates)
        for class_name, values in diagnostics.items():
            gating_rows.append({
                "seed": int(seed), "regime": regime, "map": map_name,
                "class": class_name,
                "map_reliable": bool(reliable),
                "gate_defect": values["gate_defect"],
                "heldout_defect": values["heldout_defect"],
                "gradient_cosine": values["gradient_cosine"],
                "accepted": float(values["accepted"]),
                "gate": values["gate"],
            })

    learned_map = seed_payload["maps"]["learned"][0]
    shuffled_train = _action_pairs(
        source_operator, learned_map, seed_payload["train_probes"],
        args.action_horizons, shuffled=True, generator=generator
    )
    shuffled_test = _action_pairs(
        source_operator, learned_map, seed_payload["test_probes"],
        args.action_horizons, shuffled=True, generator=generator
    )
    shuffled_gate = _action_pairs(
        source_operator, learned_map, seed_payload["gate_probes"],
        args.action_horizons, shuffled=True, generator=generator
    )
    shuffled_constraints = {
        name: {
            "train": shuffled_train[name], "gate": shuffled_gate[name],
            "test": shuffled_test[name]
        }
        for name in shuffled_train
    }
    shuffled_gates, shuffled_diagnostics = _gate_diagnostics(
        initial_operator, task_train_pairs, shuffled_constraints,
        True, args.defect_threshold
    )
    for class_name, values in shuffled_diagnostics.items():
        gating_rows.append({
            "seed": int(seed), "regime": regime, "map": "source_time_shuffle",
            "class": class_name, "map_reliable": True,
            "gate_defect": values["gate_defect"],
            "heldout_defect": values["heldout_defect"],
            "gradient_cosine": values["gradient_cosine"],
            "accepted": float(values["accepted"]), "gate": values["gate"],
        })

    conditions = {
        "local_only": ({}, {}, False, False),
        "action_flow_ground_truth": (
            constraints_by_map["ground_truth"][0],
            constraints_by_map["ground_truth"][1], True, True
        ),
        "action_flow_learned": (
            constraints_by_map["learned"][0],
            constraints_by_map["learned"][1], True, True
        ),
        "wrong_map_gated": (
            constraints_by_map["wrong"][0],
            constraints_by_map["wrong"][1], False, False
        ),
        "identity_map_gated": (
            constraints_by_map["identity"][0],
            constraints_by_map["identity"][1], False, False
        ),
        "source_time_shuffle_gated": (
            shuffled_constraints, shuffled_gates, True, True
        ),
        "harmful_ungated": (
            constraints_by_map["ground_truth"][0],
            {"satisfied": 0.0, "transferable": 0.0, "harmful": 1.0},
            False, True
        ),
    }

    true_constraints = constraints_by_map["ground_truth"][0]
    task_before = float(_task_loss(initial_operator, task_test_pairs).item())
    condition_rows = []
    final_operators = {}
    start = time.perf_counter()
    for condition, (constraints, gates, dynamic_gates, map_reliable) in conditions.items():
        active_constraints = constraints if constraints else true_constraints
        final_operator, final_gates, active_fractions = _optimize(
            initial_operator, task_train_pairs, active_constraints, gates,
            args.steps, args.learning_rate, args.flow_strength,
            dynamic_gates=dynamic_gates, map_reliable=map_reliable,
            defect_threshold=args.defect_threshold
        )
        final_operators[condition] = final_operator
        initial_flow = _flow_gradient(initial_operator, active_constraints, gates)
        final_flow = _flow_gradient(final_operator, active_constraints, final_gates)
        row = {
            "seed": int(seed), "regime": regime, "condition": condition,
            "task_before": task_before,
            "task_after": float(_task_loss(final_operator, task_test_pairs).item()),
            "parameter_delta_norm": float(torch.linalg.norm(
                final_operator - initial_operator
            ).item()),
            "flow_norm_initial": float(torch.linalg.norm(initial_flow).item()),
            "flow_norm_final": float(torch.linalg.norm(final_flow).item()),
        }
        for class_name in ("satisfied", "transferable", "harmful"):
            row[class_name + "_gate_active_fraction"] = active_fractions.get(
                class_name, 0.0
            )
        for class_name in ("satisfied", "transferable", "harmful"):
            before = _relative_defect(initial_operator, true_constraints[class_name]["test"])
            after = _relative_defect(final_operator, true_constraints[class_name]["test"])
            row[class_name + "_defect_before"] = before
            row[class_name + "_defect_after"] = after
            row[class_name + "_defect_reduction"] = (before - after) / max(before, 1e-12)
        condition_rows.append(row)

    # Strong averaging control: map the complete source operator into receiver
    # coordinates, then take a step of the same norm as the learned action flow's
    # extra displacement beyond local-only.
    local_operator = final_operators["local_only"]
    learned_operator = final_operators["action_flow_learned"]
    transported_source = c_true @ source_operator @ c_true.T
    average_direction = transported_source - local_operator
    matched_norm = torch.linalg.norm(learned_operator - local_operator)
    parameter_average = local_operator + average_direction * (
        matched_norm / torch.linalg.norm(average_direction).clamp_min(1e-12)
    )
    row = {
        "seed": int(seed), "regime": regime,
        "condition": "norm_matched_parameter_average",
        "task_before": task_before,
        "task_after": float(_task_loss(parameter_average, task_test_pairs).item()),
        "parameter_delta_norm": float(torch.linalg.norm(
            parameter_average - initial_operator
        ).item()),
        "flow_norm_initial": 0.0,
        "flow_norm_final": 0.0,
    }
    for class_name in ("satisfied", "transferable", "harmful"):
        row[class_name + "_gate_active_fraction"] = 0.0
    for class_name in ("satisfied", "transferable", "harmful"):
        before = _relative_defect(initial_operator, true_constraints[class_name]["test"])
        after = _relative_defect(parameter_average, true_constraints[class_name]["test"])
        row[class_name + "_defect_before"] = before
        row[class_name + "_defect_after"] = after
        row[class_name + "_defect_reduction"] = (before - after) / max(before, 1e-12)
    condition_rows.append(row)

    local_after = next(
        row["task_after"] for row in condition_rows if row["condition"] == "local_only"
    )
    for row in condition_rows:
        row["task_reduction"] = (row["task_before"] - row["task_after"]) / max(
            row["task_before"], 1e-12
        )
        row["extra_gain_vs_local"] = local_after - row["task_after"]
        row["flow_decay_ratio"] = row["flow_norm_final"] / max(
            row["flow_norm_initial"], 1e-12
        )
        row["wall_time_seconds_regime"] = time.perf_counter() - start
    return gating_rows, condition_rows


def _fmt(stat, percent=False):
    scale = 100.0 if percent else 1.0
    return "{:.4f} ± {:.4f}".format(
        scale * stat["mean"], scale * stat["std"]
    )


def _write_report(path, config, summary):
    lines = [
        "# Functional Action Flow known-map pilot",
        "",
        "This is a multi-seed CPU mechanism pilot on stable six-dimensional linear systems. "
        "It is not a graph-dataset accuracy result.",
        "",
        "## Functional Map recovery",
        "",
        "| map | relative map error | held-out descriptor error | cycle error |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in summary["maps"]:
        lines.append("| {} | {} | {} | {} |".format(
            row["map"], _fmt(row["map_relative_error"]),
            _fmt(row["heldout_descriptor_error"]), _fmt(row["cycle_error"])
        ))
    lines.extend([
        "", "## Learned-map gate diagnostics", "",
        "| regime | action class | defect | task/constraint cosine | accepted |",
        "| --- | --- | ---: | ---: | ---: |",
    ])
    for row in summary["learned_gate"]:
        lines.append("| {} | {} | {} | {} | {} |".format(
            row["regime"], row["class"], _fmt(row["heldout_defect"]),
            _fmt(row["gradient_cosine"]), _fmt(row["accepted"], percent=True)
        ))
    lines.extend([
        "", "## Receiver outcomes", "",
        "| regime | condition | task reduction | extra gain vs local | transferable defect reduction | final/initial flow |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ])
    for row in summary["conditions"]:
        lines.append("| {} | {} | {} | {} | {} | {} |".format(
            row["regime"], row["condition"], _fmt(row["task_reduction"], True),
            _fmt(row["extra_gain_vs_local"]),
            _fmt(row["transferable_defect_reduction"], True),
            _fmt(row["flow_decay_ratio"]),
        ))
    decision = summary["decision"]
    lines.extend([
        "", "## Preregistered decision", "",
        "**{}**".format(decision["status"]), "",
    ])
    for item in decision["checks"]:
        lines.append("- [{}] {}".format("x" if item["passed"] else " ", item["name"]))
    lines.extend([
        "", "## Provenance", "",
        "- Seeds: `{}`".format(config["seeds"]),
        "- Git commit: `{}`".format(config["git"]["commit"]),
        "- Command: `{}`".format(config["command"]),
        "- Float dtype: `float64`; device: `CPU`",
        "- Proposed conditions perform no parameter averaging. The norm-matched parameter average is a control only.",
    ])
    path.write_text("\n".join(lines) + "\n")


def _decision(summary):
    map_stats = {row["map"]: row for row in summary["maps"]}
    gates = {
        (row["regime"], row["class"]): row for row in summary["learned_gate"]
    }
    outcomes = {
        (row["regime"], row["condition"]): row for row in summary["conditions"]
    }
    regimes = ["feature_only", "structure_only", "joint"]
    checks = [
        {
            "name": "learned map held-out descriptor error < 0.10",
            "passed": map_stats["learned"]["heldout_descriptor_error"]["mean"] < 0.10,
        },
        {
            "name": "learned map accepts transferable actions in at least 80% of seed/regime cells",
            "passed": bool(np.mean([
                gates[(regime, "transferable")]["accepted"]["mean"]
                for regime in regimes
            ]) >= 0.80),
        },
        {
            "name": "learned map accepts at most 20% of satisfied or harmful actions",
            "passed": bool(max([
                gates[(regime, class_name)]["accepted"]["mean"]
                for regime in regimes for class_name in ("satisfied", "harmful")
            ]) <= 0.20),
        },
        {
            "name": "learned action flow beats matched local-only task loss in every regime",
            "passed": all(
                outcomes[(regime, "action_flow_learned")]["extra_gain_vs_local"]["mean"] > 0.0
                for regime in regimes
            ),
        },
        {
            "name": "learned action flow reduces the true transferable defect in every regime",
            "passed": all(
                outcomes[(regime, "action_flow_learned")]["transferable_defect_reduction"]["mean"] > 0.0
                for regime in regimes
            ),
        },
        {
            "name": "gated wrong-map and identity-map controls have no positive mean gain over local-only",
            "passed": all(
                outcomes[(regime, condition)]["extra_gain_vs_local"]["mean"] <= 1e-12
                for regime in regimes
                for condition in ("wrong_map_gated", "identity_map_gated")
            ),
        },
        {
            "name": "ungated harmful transfer is worse than learned gated action flow in every regime",
            "passed": all(
                outcomes[(regime, "harmful_ungated")]["task_reduction"]["mean"]
                < outcomes[(regime, "action_flow_learned")]["task_reduction"]["mean"]
                for regime in regimes
            ),
        },
        {
            "name": "source/time-shuffled transfer is worse than learned action flow in every regime",
            "passed": all(
                outcomes[(regime, "source_time_shuffle_gated")]["task_after"]["mean"]
                > outcomes[(regime, "action_flow_learned")]["task_after"]["mean"]
                for regime in regimes
            ),
        },
        {
            "name": "dynamic gate prevents source/time shuffle from materially harming local-only",
            "passed": all(
                outcomes[(regime, "source_time_shuffle_gated")]["task_after"]["mean"]
                <= max(
                    1.10 * outcomes[(regime, "local_only")]["task_after"]["mean"],
                    outcomes[(regime, "local_only")]["task_after"]["mean"] + 1e-8,
                )
                for regime in regimes
            ),
        },
        {
            "name": "learned action flow beats the norm-matched mapped parameter average in every regime",
            "passed": all(
                outcomes[(regime, "action_flow_learned")]["task_after"]["mean"]
                < outcomes[(regime, "norm_matched_parameter_average")]["task_after"]["mean"]
                for regime in regimes
            ),
        },
        {
            "name": "cost is lower than the R010 full-response path (not evaluated in this pilot)",
            "passed": False,
            "evaluated": False,
        },
    ]
    passed = sum(item["passed"] for item in checks)
    if passed == len(checks):
        status = "SUPPORTED WITHIN THE CONTROLLED PILOT"
    elif passed >= len(checks) - 3:
        status = "PARTIALLY SUPPORTED; FAILED CHECKS REQUIRE REVISION"
    else:
        status = "NOT SUPPORTED BY THE CONTROLLED PILOT"
    return {"status": status, "passed": passed, "total": len(checks), "checks": checks}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 37, 42, 59])
    parser.add_argument("--rank", type=int, default=6)
    parser.add_argument("--descriptor-columns", type=int, default=24)
    parser.add_argument("--probe-columns", type=int, default=12)
    parser.add_argument("--task-columns", type=int, default=36)
    parser.add_argument("--action-horizons", nargs="+", type=float, default=[0.5, 1.0])
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=0.025)
    parser.add_argument("--flow-strength", type=float, default=4.0)
    parser.add_argument("--map-error-threshold", type=float, default=0.10)
    parser.add_argument("--defect-threshold", type=float, default=0.01)
    parser.add_argument("--output-root", type=Path, default=PROJECT / "run_logs")
    parser.add_argument("--run-tag", default="functional_intertwining_r012")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.rank != 6:
        raise ValueError("The preregistered action classes require --rank 6.")
    torch.set_default_dtype(torch.float64)
    timestamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output = args.output_root / "{}_{}".format(args.run_tag, timestamp)
    output.mkdir(parents=True, exist_ok=False)

    config = vars(args).copy()
    config["output_root"] = str(config["output_root"])
    config["git"] = _git_state()
    config["command"] = " ".join(os.sys.argv)
    config["torch_version"] = torch.__version__
    _json_dump(output / "config.json", config)

    map_rows = []
    gating_rows = []
    condition_rows = []
    start = time.perf_counter()
    for seed in args.seeds:
        seed_payload = _build_seed(
            seed, args.rank, args.descriptor_columns, args.probe_columns,
            args.task_columns
        )
        map_rows.extend(seed_payload["map_rows"])
        for regime in ("feature_only", "structure_only", "joint"):
            gates, conditions = _run_regime(seed_payload, seed, regime, args)
            gating_rows.extend(gates)
            condition_rows.extend(conditions)

    local_lookup = {
        (row["seed"], row["regime"]): row["task_after"]
        for row in condition_rows if row["condition"] == "local_only"
    }
    for row in condition_rows:
        row["extra_gain_vs_local"] = (
            local_lookup[(row["seed"], row["regime"])] - row["task_after"]
        )

    summary = {
        "maps": _aggregate(
            map_rows, ["map"],
            ["map_relative_error", "heldout_descriptor_error", "cycle_error"]
        ),
        "learned_gate": _aggregate(
            [row for row in gating_rows if row["map"] == "learned"],
            ["regime", "class"],
            ["heldout_defect", "gradient_cosine", "accepted", "gate"]
        ),
        "conditions": _aggregate(
            condition_rows, ["regime", "condition"],
            [
                "task_before", "task_after", "task_reduction",
                "extra_gain_vs_local", "transferable_defect_reduction",
                "harmful_defect_reduction", "parameter_delta_norm",
                "flow_norm_initial", "flow_norm_final", "flow_decay_ratio",
                "satisfied_gate_active_fraction",
                "transferable_gate_active_fraction",
                "harmful_gate_active_fraction",
            ]
        ),
        "communication": {
            "bytes_per_source_calibration_float64": int(
                args.rank * args.rank * 8
                + 3 * args.rank * args.probe_columns
                * (1 + len(args.action_horizons)) * 8
            ),
        },
        "wall_time_seconds": time.perf_counter() - start,
    }
    summary["decision"] = _decision(summary)

    with (output / "map_records.jsonl").open("w") as handle:
        for row in map_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with (output / "gating_records.jsonl").open("w") as handle:
        for row in gating_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with (output / "condition_records.jsonl").open("w") as handle:
        for row in condition_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    _json_dump(output / "summary.json", summary)
    _write_report(output / "REPORT.md", config, summary)

    print(str(output))
    print(summary["decision"]["status"])
    print("checks: {}/{}".format(
        summary["decision"]["passed"], summary["decision"]["total"]
    ))
    print("wall_time_seconds: {:.3f}".format(summary["wall_time_seconds"]))


if __name__ == "__main__":
    main()
