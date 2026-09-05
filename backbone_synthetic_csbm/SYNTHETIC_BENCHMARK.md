# Controlled Synthetic GFL Benchmark

## Goal

The default task is now a synthetic federated graph benchmark with 10 clients, 1000 nodes per client, 128-dimensional node features, and 8 classes. Client 0 is always the reference client. For client `m`, the ground-truth heterogeneity severity is:

$$
\lambda_m = \frac{m}{M-1}.
$$

The generator is a multi-class contextual stochastic block model:

$$
y_i \sim \pi_m,
$$

$$
x_i\mid y_i=c \sim \mathcal N(\mu_{m,c},\sigma_m^2I),
$$

$$
A_{ij}\mid y_i=c,y_j=c' \sim \mathrm{Bernoulli}(B_{m,cc'}).
$$

The graph component uses NetworkX's mature `stochastic_block_model` implementation. The generator solves `p_in` and `p_out` from target average degree and target edge homophily, so feature-only experiments can keep the expected graph statistics fixed even when class proportions change.

## Scenarios

| Scenario | Label proportion | Class-conditional Gaussian | Homophily | Average degree |
| --- | --- | --- | --- | --- |
| `iid` | fixed | fixed | fixed | fixed |
| `label_shift` | increasing shift | fixed | fixed | fixed |
| `feature_shift` | fixed | increasing shift | fixed | fixed |
| `feature_mixed` | increasing shift | increasing shift | fixed | fixed |
| `structure_homophily` | fixed | fixed | increasing shift | fixed |
| `structure_degree` | fixed | fixed | fixed | increasing shift |
| `structure_mixed` | fixed | fixed | increasing shift | increasing shift |
| `mixed` | increasing shift | increasing shift | increasing shift | increasing shift |

The default scenario is `feature_shift`, because the current research stage focuses on initial-state feature heterogeneity.

The default difficulty is calibrated so the official A-DGN + FedAvg baseline remains below 90% after 100 rounds. Relative to the original easy setting, class means are closer, feature noise is larger, and graph homophily is lower:

```text
class separation = 2.3
feature std = 1.7
maximum feature mean shift = 2.5
average degree = 12
edge homophily = 0.65
```

With seed 42, the calibrated 100-round `feature_shift` baseline reaches `88.68 ± 2.68%` client-paired test accuracy.

## Default Run

No dataset argument is required anymore:

```bash
python main.py --base-path /opt/data/private/xzc/work2
```

Equivalent explicit form:

```bash
python main.py \
  --dataset Synthetic \
  --synthetic-scenario feature_shift \
  --n-clients 10
```

For another controlled regime:

```bash
python main.py --dataset Synthetic --synthetic-scenario structure_homophily
```

To run all regimes:

```bash
bash scripts/run_synthetic_all.sh
```

Use `--synthetic-regenerate` after changing generation hyperparameters. Use `--synthetic-no-viz` only when dataset plots are not needed.

## Dataset Diagnostics

Synthetic data are cached under:

```text
datasets/Synthetic_<scenario>_disjoint/10/
```

Each generated benchmark contains:

- `synthetic_summary.csv`: exact and realized heterogeneity measurements for all clients;
- `metadata.json`: complete generator configuration;
- `diagnostics/01_label_proportions.png`;
- `diagnostics/02_feature_pca_by_client.png`;
- `diagnostics/03_feature_pca_reference_vs_last.png`;
- `diagnostics/03b_feature_class_centroid_shift.png`;
- `diagnostics/04_average_degree.png`;
- `diagnostics/05_homophily.png`;
- `diagnostics/06_heterogeneity_components.png`;
- adjacency visualizations for client 0 and the maximally heterogeneous client.

The summary includes:

- ground-truth severity;
- label Jensen-Shannon divergence to client 0;
- analytic class-conditional Gaussian Wasserstein-2 distance to client 0;
- target and realized average degree;
- target and realized edge homophily;
- degree coefficient of variation;
- all class counts and proportions.

## Training Diagnostics

After a synthetic training run, the normal log directory additionally contains:

- `synthetic_performance_summary.csv`;
- `synthetic_accuracy_vs_severity.png`;
- `synthetic_accuracy_vs_feature_w2.png` when feature shift is active.
- `initial_state/initial_state_before_pca_grid.png` for per-client Graph ODE initial-state distributions;
- `initial_state/initial_state_before_pca_all_clients.png` using one shared PCA coordinate system;
- `initial_state/initial_state_before_class_centroid_trajectories.png`;
- `initial_state/initial_state_before_distance_to_client0.png` and its summary CSV.

Initial-state snapshots are collected before local training at round 0, after every client receives the same global model. They are local diagnostics only and are not included in client-to-server messages.

This makes it possible to directly inspect whether client accuracy or future heterogeneity-correction strength changes monotonically with known ground-truth heterogeneity.

## Important Design Choice

For `label_shift` and `feature_mixed`, the realized class proportions change across clients. The SBM probabilities are re-solved from the realized block sizes so that target average degree and target homophily remain fixed. This avoids accidentally introducing structural heterogeneity into a nominally feature-only experiment.
