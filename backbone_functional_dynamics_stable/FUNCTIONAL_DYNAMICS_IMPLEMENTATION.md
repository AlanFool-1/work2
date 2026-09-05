# Stable Functional Dynamics implementation

This directory is an independent copy of `backbone_synthetic_csbm`. The source
project is not modified. The implementation follows
`../FEDERATED_FUNCTIONAL_DYNAMICS_STABLE_REVISION.md` and is opt-in.

## Compatibility

Without `--enable-functional-dynamics`, model construction and training retain
the official A-DGN forward and the existing FedAvg lifecycle. With the flag,
the A-DGN Euler field receives the cached low-rank correction described in the
revision. Round-end calibration always disables this correction and runs one
no-gradient native forward.

## Online bootstrap

The multiprocess runtime has one client-to-server exchange per communication
round, so bootstrap is implemented without an extra synchronization barrier:

1. first calibration uploads deterministic-gauge descriptors and generators;
2. the server establishes the first canonical descriptor aggregate;
3. next calibration initializes each map using descriptor-only Procrustes;
4. subsequent calibrations optimize descriptor and dynamics residuals jointly;
5. injection starts only after a descriptor-initialized map is cached.

Thus no dynamics term or functional injection is used before descriptor
bootstrap is complete.

## Run

```bash
source activate torch
cd /opt/data/private/xzc/work2/backbone_functional_dynamics_stable
python main.py \
  --dataset Synthetic \
  --n-clients 10 --n-workers 10 --n-rnds 50 \
  --gpu 0,1 --enable-functional-dynamics \
  --run-tag functional_dynamics_stable
```

The default method hyperparameters match the revision: rank 8, probe dimension
4, ridge coefficient `1e-3`, 10 Functional Map steps, beta `0.05`, spectral
limit `1.0`, and EMA coefficient `0.9` for both canonical summaries. All can be
overridden through the `--fd-*` CLI options.

`functional_dynamics.csv` records generator fit, Functional Map residuals,
basis staleness, correction stability, and injection/native norm ratio.
At successful completion, these measurements are rendered to
`functional_dynamics_diagnostics.png`; the final canonical generator spectrum
and descriptor are rendered to `canonical_functional_dynamics.png`.
Diagnostic-profile runs also render every client's round-wise validation
accuracy and their mean to `client_validation_accuracy.png`.

The `federation` component is explicitly a **simulated secure aggregation**
boundary. It does not implement a cryptographic protocol and must not be
described as one.

## Multi-regime prototypes

Set `--fd-num-prototypes 2` (or more) to cluster normalized trajectory
descriptors using gauge-invariant orthogonal-Procrustes distances. Bootstrap
uses deterministic K-medoids, enforces `--fd-min-cluster-size`, aligns every
member to its cluster medoid before averaging, and records that this initial
step is centralized diagnostic clustering. Clients subsequently select their
prototype locally, with `--fd-cluster-switch-margin` hysteresis. Cluster IDs,
distances, affinities, and switches are logged; assignments are rendered to
`functional_cluster_assignments.png`.

`--fd-map-type regularized` enables the general (non-orthogonal) Functional
Map used by the multi-regime variant. It minimizes scale-normalized descriptor
and dynamics residuals with backtracking, bounded singular values, ridge
regularization, and an optional near-orthogonality penalty. Generator transport
then uses the similarity transform `C @ A @ inverse(C)` and pullback uses
`inverse(C) @ A_star @ C`; the orthogonal mode continues to use `C.T`.

`--fd-normalize-generator` separates generator speed from shape:
`A_scale = ||A||_F`, `A_map = A / A_scale`. Functional Maps and prototypes
operate on the dimensionless shape, while pullback restores the receiving
client's own speed before constructing the correction. This prevents absolute
trajectory rates from being transferred between heterogeneous clients.

The validated Cora/10-client settings are stored in `configs/s0_ode_gnn.json`.
They are applied only when the functional-dynamics feature flag is enabled;
the corresponding results must be reported using each client's
best-validation checkpoint and its paired test score, never the final round.

## Verification

```bash
source activate torch
python -m unittest discover -s tests -p 'test_*.py' -v
python tests/validate_s0.py
```

Tests cover deterministic orthonormal bases, ridge fitting, orthogonal-map
constraints, dissipative/spectral projection, gradient flow through the fixed
injection, weighted aggregation with EMA, and exact native-backbone equivalence
when the feature is disabled.
