# FedDGN Functional-Map server stage

This package implements the first stage described by
`Koopman_Guided_Federated_Graph_Dynamics_v0.md` and
`feddgn_local_koopman_visualization.md`, followed by the passive exchange in
`FedDGN_modeGraph_v0_design.md` and
`FedDGN_FunctionalMap_Server_Algorithm.md`.

The task path uses the unchanged official-style 16-step A-DGN architecture
from `models/feddgn_local`, but its network parameters now follow FedAvg.  At
the start of every round each client installs the server average and uses the
O4 `block_v` protocol: server-aggregated blockwise Adam second moments with a
fresh local first moment.  The trainable part of the ResKoopNet dictionary is
averaged through a separate parameter FedAvg.  After installing that global
dictionary, each client rebuilds a fresh Koopman Adam; no Koopman optimizer
moments are transported.  After the task update, a second detached rollout is
converted to adjacent node-time snapshot pairs. The dictionary is trained by
spectral residuals and the client-specific K is recomputed analytically by
ridge EDMD.

After each local update, every client uploads its small Koopman operator and
its best finite non-constant Koopman modes.  The server selects the client
with the lowest spectral residual as the shared reference and solves

`min_C ||C K_m - K_s C||_F`, subject to `C^H C = I`.

The solver uses spectrum-matched polar initialization followed by a few
unitary projected-gradient refinements.  It acts only on the small observable
operator (81 by 81 by default).  Each local Koopman eigenvector is transported
as `v_tilde = C_m v`; raw vectors from independently learned observable
coordinates are never compared directly.  The server then uses one edge
signal,

`W_ij = |<v_tilde_i / ||v_tilde_i||, v_tilde_j / ||v_tilde_j||>|`.

Eigenvalues and residuals remain diagnostic metadata and never change an edge
weight.  Edges are cross-client only.  For each mode, the server first keeps
the single best spatial match from every foreign client, then retains the best
five distinct-client candidates and symmetrizes their union.  Deterministic
Louvain discovers the number of communities directly, without eigengap or a
fixed K.  Each community produces a phase-aligned representative and mean
eigenvalue, forming the global mode bank.  Each client receives its own
Functional Map, the global bank, and its own modes plus related cross-client
modes.  The response remains analysis-only, not an averaged operator.

The Koopman/Functional-Map branch remains deliberately task-passive:

- compact Koopman modes, the small K, and the trainable dictionary parameters
  are uploaded; raw trajectories remain private;
- dictionary parameters are averaged, while each analytic K remains local;
- Koopman Adam state is reset after a global dictionary receive and is never
  aggregated;
- no Koopman operator is aggregated;
- no generator or external force is constructed;
- no ResKoopNet gradient reaches the A-DGN model;
- returned mode banks are not installed in the vector field;
- the server's task-model aggregate is installed normally through FedAvg, but
  no Koopman quantity changes that aggregation or the local task forward pass.

The server also reports full-space Silhouette, within/between distance ratio,
modularity, and mean conductance.  Thus a visually partitioned sparse graph is
not mistaken for well-separated communities in the aligned geometry.

Each client writes the latest saved analysis under
`<run>/koopman/client_<id>/`:

- `K.pt`
- `eigenvalues.pt`
- `eigenvectors.pt`
- `residual.pt` (one normalized residual per eigenmode)
- `modes.pt` (state-space modes, shape `state_dim x observable_dim`)
- `dictionary.pt`
- `metadata.json`
- `personalized_modes.pt` / `personalized_modes.json` (latest server response)

The server writes the latest full graph under
`<run>/koopman/mode_graph/latest.pt` and a JSON global view under
`latest.json`.  Rounds selected by `feddgn_mode_graph_save_every` (and the
final round) are archived as `round_XXXX/`, including one tensor view per
client.  Every client view contains local modes and a spatially ranked,
community-constrained bank of related modes from other clients.

At every archived mode-graph round the server creates an image dashboard under
`<run>/koopman/mode_graph/round_XXXX/figures/index.html`.  With the default
`feddgn_mode_graph_save_every=10`, rendering happens only at rounds 10, 20,
..., 100, never on every training round.  Each dashboard contains the global
mode network, full aligned-similarity heatmap, client-community count matrix,
normalized client community distribution, Functional Map
residual/orthogonality diagnostics, and community evolution
across archived graph rounds.  This is controlled by
`feddgn_mode_graph_visualize` independently from the older client-local
Koopman figures.

At finalization the server renders spectrum, magnitude/decay, reconstruction
residual, client PCA embedding, phase-invariant mode similarity, operator
heatmaps, and reliability-weighted mode energy under
`<run>/koopman/figures/`.

Example smoke run:

```bash
python main.py \
  --model feddgn \
  --dataset Cora \
  --config configs/feddgn.json \
  --n-clients 10 \
  --n-workers 10 \
  --n-rnds 2 \
  --n-eps 1 \
  --gpu 0,1
```

The implementation follows the algorithmic structure of the public
ResKoopNet ICML 2025 code (learned dictionary, analytic K, frozen eigensystem
during inner dictionary updates), while using the repository's checkpointing,
device, and federated lifecycle conventions.
