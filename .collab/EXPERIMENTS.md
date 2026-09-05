# Experiments

Updated: 2026-09-05 UTC

Raw logs remain under `logs/` and code-local `run_logs/`. This file stores only evidence needed for research decisions.

## E001 - Official A-DGN + FedAvg real-dataset baseline

### Configuration

- 10 clients, equal FedAvg, persistent Adam.
- Official A-DGN with 16 Euler steps.
- Selection: each client's best-validation round paired with its test metric.
- Run tag: `official_adgn_k16_all11_20260903_022044`.

### Results

The authoritative table is `backbone_functional_dynamics_stable/BASELINE_RESULTS.json`. It covers Cora, CiteSeer, PubMed, Computers, Photo, ogbn-arxiv, Roman-empire, Amazon-ratings, Minesweeper, Tolokers, and Questions.

### Decision use

Use these values as the baseline for E003 and later matched real-dataset comparisons. Do not compare against final-round metrics.

## E002 - Controlled synthetic feature-shift baseline

### Configuration

- Seed 42, 10 clients, about 1,000 nodes per client, 128 features, 8 classes.
- Feature shift increases monotonically with client index while graph statistics are held fixed.
- 100 communication rounds.

### Result

Official A-DGN + FedAvg paired test accuracy: `88.68 +/- 2.68%`.

### Evidence

See `backbone_functional_dynamics_stable/SYNTHETIC_BENCHMARK.md` for the generator and diagnostic contract.

### Decision use

Use this benchmark for controlled mechanism tests that must separate feature-side effects from structural heterogeneity.

## E003 - Multi-prototype Functional Dynamics on 11 real datasets

### Configuration

- Seed 42, 10 clients, model `s0_ode`, equal aggregation, persistent Adam.
- Dataset-specific round and model settings come from `backbone_functional_dynamics_stable/configs/s0_ode_gnn.json` and each run's `config.json`.
- Per-client best-validation paired test selection.
- Cora, CiteSeer, PubMed, and Computers came from run tag `functional_dynamics_multi_proto_all11_20260905_013318`.
- The remaining datasets came from resumed run tag `functional_dynamics_multi_proto_all11_20260905_023525`.

### Results

Values are percentages. Delta is Functional Dynamics minus E001 in percentage points.

| Dataset | Metric | Functional Dynamics | Official baseline | Delta |
| --- | --- | ---: | ---: | ---: |
| Cora | ACC | 80.8828 | 80.2680 | +0.6148 |
| CiteSeer | ACC | 76.3494 | 74.9667 | +1.3827 |
| PubMed | ACC | 86.5571 | 86.6107 | -0.0536 |
| Computers | ACC | 84.9982 | 84.2061 | +0.7921 |
| Photo | ACC | 90.6232 | 89.2568 | +1.3664 |
| ogbn-arxiv | ACC | 66.4181 | 66.3899 | +0.0282 |
| Roman-empire | ACC | 70.5185 | 70.4883 | +0.0302 |
| Amazon-ratings | ACC | 42.0171 | 41.8015 | +0.2156 |
| Minesweeper | AUC | 87.1962 | 87.2255 | -0.0293 |
| Tolokers | AUC | 74.3916 | 74.6877 | -0.2961 |
| Questions | AUC | 68.4084 | 68.7566 | -0.3482 |

### Evidence

- First four: `logs/{Cora,CiteSeer,PubMed,Computers}_disjoint/clients_10/*functional_dynamics_multi_proto_all11_20260905_013318*/result.json`.
- Remaining seven: `backbone_functional_dynamics_stable/run_logs/functional_dynamics_multi_proto_all11_20260905_023525/status.tsv` and the referenced `result.json` files.
- Mechanism diagnostics: `functional_dynamics.csv`, `functional_dynamics_diagnostics.png`, `functional_cluster_assignments.png`, and canonical generator plots in each result directory.

### Interpretation

This is preliminary single-seed evidence. Gains are concentrated in a few datasets, while several results are effectively unchanged or slightly lower. The matrix supports further diagnosis but does not yet establish that cross-client dynamics transfer or clustering causes the gains.

The first run's `status.tsv` labels Computers as failed with exit code 0 even though a complete `result.json` exists. Audit the orchestration record separately from the model result.

### Decision

Set the current handoff to `NEEDS_RESEARCH`. Analyze mechanism diagnostics and define matched controls before expanding implementation or tuning.
