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

- Seed 42, 10 clients, 2 local epochs, model `s0_ode`, equal aggregation, persistent Adam.
- Three prototypes, minimum cluster size 2, switch margin 1.0, rank 16, probe dimension 4.
- Regularized Functional Maps with condition limit 20, orthogonality regularization 0.01, and 20 solver steps.
- Generator normalization enabled; injection coefficient 0.05; dissipative spectral limit 1.0; prototype EMA 0.9.
- Computers, Photo, and ogbn-arxiv use 200 rounds and hidden dimension 128. Other datasets use 100 rounds and hidden dimension 64.
- Dataset-specific settings come from `backbone_functional_dynamics_stable/configs/s0_ode_gnn.json` and each run's `config.json`.
- Per-client best-validation paired test selection.
- Cora, CiteSeer, PubMed, and Computers came from run tag `functional_dynamics_multi_proto_all11_20260905_013318`.
- The remaining datasets came from resumed run tag `functional_dynamics_multi_proto_all11_20260905_023525`.

### Run integrity

- The selected result directory for every dataset contains nonempty `result.json`, `config.json`, `client_best.csv`, `functional_dynamics.csv`, and final diagnostic plots.
- The first Photo attempt (`20260905_015711...Photo`) stopped after training records and is excluded because it has no `result.json` or final plots. The completed resumed Photo run (`20260905_023528...Photo`) is used below.
- The first orchestration directory labels Computers as failed with exit code 0 and lost its dataset log, but the corresponding result directory has a complete 200-round result and diagnostics. Treat this as stale orchestration metadata rather than a failed model run.

### Results

Values are percentages. Main result and F1 are client mean +/- client standard deviation. Delta is Functional Dynamics minus E001 in percentage points.

| Dataset | Metric | Functional Dynamics | Official baseline | Delta | F1 | Rounds |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Cora | ACC | 80.8828 +/- 10.4796 | 80.2680 | +0.6148 | 49.3564 +/- 9.8118 | 100 |
| CiteSeer | ACC | 76.3494 +/- 10.4005 | 74.9667 | +1.3827 | 39.3494 +/- 12.7655 | 100 |
| PubMed | ACC | 86.5571 +/- 3.3280 | 86.6107 | -0.0536 | 73.7244 +/- 9.7664 | 100 |
| Computers | ACC | 84.9982 +/- 9.3596 | 84.2061 | +0.7921 | 37.5802 +/- 18.1089 | 200 |
| Photo | ACC | 90.6232 +/- 4.7128 | 89.2568 | +1.3664 | 37.5956 +/- 12.5760 | 200 |
| ogbn-arxiv | ACC | 66.4181 +/- 6.6809 | 66.3899 | +0.0282 | 26.3337 +/- 5.0977 | 200 |
| Roman-empire | ACC | 70.5185 +/- 2.4893 | 70.4883 | +0.0302 | 61.7117 +/- 2.5931 | 100 |
| Amazon-ratings | ACC | 42.0171 +/- 5.0984 | 41.8015 | +0.2156 | 21.7966 +/- 6.2328 | 100 |
| Minesweeper | AUC | 87.1962 +/- 3.1409 | 87.2255 | -0.0293 | 52.3059 +/- 9.0774 | 100 |
| Tolokers | AUC | 74.3916 +/- 6.4703 | 74.6877 | -0.2961 | 34.1406 +/- 23.8265 | 100 |
| Questions | AUC | 68.4084 +/- 4.8263 | 68.7566 | -0.3482 | 9.1451 +/- 12.5414 | 100 |

Seven of 11 point estimates are positive and four are negative. The descriptive mean delta is `+0.3366` points and the median is `+0.0302` points; only four datasets exceed `+0.5` points, and none is below `-0.5` points. Because ACC and AUC are mixed and only one seed was run, the cross-dataset mean is not a statistical claim of improvement.

### Steady-state mechanism diagnostics

The following values are medians over the last 20% of rounds. `Basis overlap` is the logged `basis_staleness` field; higher values mean a more stable consecutive-round subspace. Injection is the injected/native field norm ratio in percent. Clip scale below 1 means the spectral limit is active.

| Dataset | Generator fit error | Basis overlap | Final FM objective | Map condition | Injection (%) | Clip scale | Final cluster sizes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Cora | 0.3606 | 0.9745 | 0.02983 | 5.701 | 1.152 | 0.229 | 6 / 2 / 2 |
| CiteSeer | 0.2764 | 0.9068 | 0.02650 | 5.421 | 0.907 | 0.237 | 3 / 2 / 5 |
| PubMed | 0.2775 | 0.9881 | 0.01578 | 5.238 | 0.830 | 0.376 | 5 / 3 / 2 |
| Computers | 0.5732 | 0.9350 | 0.03443 | 3.863 | 0.677 | 0.131 | 6 / 2 / 2 |
| Photo | 0.6245 | 0.9028 | 0.03720 | 5.217 | 1.039 | 0.102 | 6 / 2 / 2 |
| ogbn-arxiv | 0.6806 | 0.9794 | 0.01560 | 4.545 | 1.321 | 0.869 | 5 / 2 / 3 |
| Roman-empire | 0.7081 | 0.9984 | 0.00861 | 6.030 | 0.572 | 1.000 | 6 / 2 / 2 |
| Amazon-ratings | 0.4372 | 0.9970 | 0.01334 | 6.328 | 0.726 | 0.737 | 5 / 3 / 2 |
| Minesweeper | 0.6872 | 0.9994 | 0.01402 | 5.758 | 0.896 | 0.485 | 4 / 2 / 4 |
| Tolokers | 0.7267 | 0.9983 | 0.01797 | 5.443 | 1.075 | 0.402 | 4 / 2 / 4 |
| Questions | 0.3713 | 0.9922 | 0.01158 | 4.731 | 0.857 | 0.543 | 5 / 3 / 2 |

### Mechanism findings

1. The Functional Map optimization is numerically active and conditioned: the median accepted-step count is 20/20 on every dataset, median objective reduction ranges from 4.67% to 28.35%, and median map conditions remain 3.86-6.33, well below the configured limit of 20.
2. Consecutive-round trajectory subspaces are stable after convergence: median overlap is 0.903-0.999.
3. The effective correction is small: the median injected field is only 0.57%-1.32% of the native field.
4. The dissipative spectral cap is usually binding. Ten of 11 datasets have a median clip scale below 1; the projected spectral norm stays at or below 1 up to numerical tolerance, and the largest observed symmetric eigenvalue is about `1.5e-7`.
5. No client changes prototype in any of the 11 runs. With switch margin 1.0, the current rule requires essentially a 100% relative objective improvement to leave the previous prototype. Since objectives are positive, the bootstrap clusters are effectively frozen. This experiment therefore evaluates fixed bootstrap regimes, not adaptive regime tracking.
6. Generator fit quality varies substantially (`0.276-0.727`) and does not visibly align with the accuracy/AUC gains. The strongest gains occur both with relatively low error (CiteSeer) and high error (Photo), while several high-error datasets are unchanged or worse.
7. These diagnostics show numerical stability, but they do not establish useful cross-client causal transfer. A local-only, shuffled-prototype, wrong-prototype, and module-off comparison is still required.

### Evidence

- First four: `logs/{Cora,CiteSeer,PubMed,Computers}_disjoint/clients_10/*functional_dynamics_multi_proto_all11_20260905_013318*/result.json`.
- Remaining seven: `backbone_functional_dynamics_stable/run_logs/functional_dynamics_multi_proto_all11_20260905_023525/status.tsv` and the referenced `result.json` files.
- Mechanism diagnostics: `functional_dynamics.csv`, `functional_dynamics_diagnostics.png`, `functional_cluster_assignments.png`, and canonical generator plots in each result directory.

### Verification on 2026-09-05

- `python -m unittest discover -s tests -p 'test_*.py' -v`: 18/18 passed.
- `python tests/validate_s0.py`: passed official-source, finite-gradient, shared-field Euler, variable-topology, and persistent-Adam invariants.

### Decision

Keep the current handoff at `NEEDS_RESEARCH`. The next research step should first decide whether to test genuinely adaptive prototypes with a meaningful switch threshold or simplify the method, then specify matched controls that isolate cross-client transfer from regularization and local dynamics effects.
