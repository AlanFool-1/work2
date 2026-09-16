# FedDGN Method V1.0: four-stage passive Koopman loop

`--model feddgn` is now the local-backbone/observable-dictionary-FedAvg
variant. The former K-star energy-force feedback (stage 5) was removed pending
a new algorithm. The old mode-graph, server-spectrum and force visualizations
were removed. Only compact, **within-client** K versus K-star diagnostics are
retained; there is no cross-client comparison chart.

1. **Local task dynamics.** Each client owns its 16-step anti-symmetric A-DGN
   encoder/field/classifier and persistent task Adam. Task parameters and
   optimizer state are never uploaded, averaged, or broadcast.
2. **Local ResKoopNet.** After task training, a detached local trajectory fits
   the learnable observable dictionary and analytic 81×81 K (default width).
   Only dictionary weights receive FedAvg. The local K, compact reliability
   metadata, and dictionary weights are uploaded; trajectories and sampled
   observable snapshots remain private.
3. **Server modes.** Small Functional Maps align client K coordinates to a
   spectral-medoid reference. Overlap-conditioned dynamical conflict screening
   returns a candidate mode package to each participating client. There is no
   global K, task-model aggregation, mode graph, or community detection.
4. **Passive K-star reconstruction.** At the next receive, the client uses the
   previous-round private K/Z/Z+ and matching candidate package to compute
   compatibility×need weights and a closed-form K-star. It does this **before**
   installing the averaged observable dictionary, then saves K-star only for
   diagnosis. K-star has no path into the task or ResKoop forward pass.

The server result/checkpoint explicitly marks `task_parameter_aggregation`,
`task_optimizer_aggregation`, `K_aggregation`, and `K_star_task_feedback` as
false. A client upload contains no `model`, task Adam moments, or O4 block-v.

## Per-round timing

`<run>/timing.csv` is written every round, independently of the logging
profile. It records these non-overlapping measured intervals:

- client `task_seconds`: A-DGN local epoch(s), validation/test, and best-model
  checkpoint;
- client `reskoop_seconds`: detached rollout, ResKoop fit/K identification,
  server packet, private snapshots, and compact numeric checkpoint;
- client `kstar_seconds`: previous-round candidate receive, private need/K-star
  solve, numeric artifact, and selective per-client figures; zero in round 1;
- client `kstar_compute_seconds`: the private K-star solve alone;
- client `kstar_visualization_seconds`: one matrix plot and the client-local HTML
  dashboard when the selected round is archived, otherwise zero;
- client `dictionary_receive_seconds`: installation of the sole FedAvg
  dictionary and fresh Koopman Adam;
- server `server_modes_build_seconds`: Functional Map, mode extraction, and
  conflict-screened candidate construction;
- server `server_modes_return_seconds`: Manager-dict broadcast of candidate
  packages and averaged dictionary;
- separate server dictionary FedAvg, numeric-artifact, visualization,
  metrics-write, and checkpoint durations, plus complete round wall time.

Client stage means and maxima are across parallel workers and must **not** be
added to obtain round wall time. `client_parallel_wall_seconds` includes worker
dispatch/load/communication while the server waits; `client_other_seconds`
shows client switch/load and other work not in the four named stages.

`<run>/koopman/candidates/latest.pt/json` and `server_rounds.csv` keep numeric
mode/quality records. `<run>/koopman/client_<id>/personalized/latest.pt/json`
keeps K-star and candidate diagnostics. Every
`feddgn_koopman_save_every` rounds (and at the final round), a compact
`latest_K.pt` keeps the current local K/spectrum/residuals. The active client
also archives its own K/K-star plots at the first eligible receive, every 10
rounds by default, and the final receive. Each archived figure compares **one
client's** K and K-star in the same previous-round private observable frame:
`<run>/koopman/client_<id>/personalized/figures/source_round_<r>/index.html`.
`<run>/koopman/client_<id>/personalized/figures/latest.html` points to that
client's latest dashboard. It contains shared-scale `log10|K|` and
`log10|K*|`, separately scaled `log10|K*-K|`, compatibility/need, candidate
funnel, private residual, active-mode residual, and round-wise knowledge-flow
plots. The server writes Functional Map quality, mode correspondence, conflict,
candidate-funnel, and history plots under
`<run>/koopman/visualization/server/`. There is **no spectrum or mode-graph
plot**. Use the client/server visualization cadence and DPI options to control
the archive cost; a cadence of 0 disables server images while preserving its
numeric logs.

For a durable Cora 10-client, 100×1 timing campaign, use
`run_feddgn_v10_local_timing100.sh`. It caps BLAS/OpenMP threads per process;
this changes resource scheduling, not the model/aggregation protocol. The
previous force-control campaign was paused at round 98 and is a historical
process with modules already loaded; do not assume it can be safely resumed
against this revised source tree.
