# Experiments

Updated: 2026-09-06 UTC

## E016 - Preregistered low-dimensional dynamics operator selection

Date: 2026-09-06 UTC. Study design only; no new numeric experiment or model
training was run.

The current proxy-selection study is specified in
`.collab/LOW_DIMENSIONAL_DYNAMICS_OPERATOR_SELECTION.md` and preregistered in
`.collab/operator_proxy_pilot.yaml`. It compares autonomous and affine
generators, delay-AR/Koopman, a low-rank finite-horizon causal response
operator, and an uncompressed response upper bound under matched rank and
communication budgets.

The decisive Stage A uses known shared and client-exclusive dynamical modes.
It tests held-out port/time response prediction, bootstrap subspace stability,
exclusive-mode recovery, and receiver missing-response repair. Stage B then
uses frozen feature, structure, and joint synthetic A-DGN checkpoints and
matched local-only, true-source, shuffled-source/time, redundant-source,
raw-response, and no-flow forks. The study is designed to reject proxies that
fit their own trajectory but cannot identify or transfer receiver-missing
dynamical capability.

No proxy is selected by evidence yet. The low-rank causal response operator is
the leading hypothesis because it has shared input-output semantics and admits
receiver-specific subspace innovation. Production source changes and the
11-dataset matrix remain deferred.

## E015 - Retrospective early-to-late flow-proxy analysis

Date: 2026-09-06 UTC. Retrospective analysis of the completed E003 runs; no new
training or checkpoint selection.

For each of the 11 complete runs, excluded startup rounds with zero injection,
then compared client-round medians in the first and last 20% of active rounds.
The reported aggregate is the median of the 11 dataset-level values or ratios.
Reproduction command:

```bash
python3 backbone_functional_dynamics_stable/scripts/analyze_flow_decay.py
```

| Old-method proxy | Datasets declining | Median early | Median late | Median late/early ratio |
| --- | ---: | ---: | ---: | ---: |
| Raw correction norm | 11/11 | 10.176 | 4.349 | 0.427 |
| Injected/native ratio | 10/11 | 1.203% | 0.896% | 0.730 |
| Post-projection safe norm | 1/11 | 1.250 | 1.492 | 1.189 |
| Cluster distance | 6/11 | 0.735 | 0.663 | 0.972 |

The raw correction and relative injection usually weaken with training, which
is compatible with decreasing external correction demand. This is not evidence
that complementary knowledge was absorbed: parameter scaling, fixed
prototypes, Functional Map evolution, optimization convergence, and clipping
can create the same pattern. In particular, the applied safe correction does
not usually decline, and no same-checkpoint local-only counterfactual exists.

Decision use: historical side evidence only. Do not use these proxies to select
a dynamics operator, or as an information-gain, absorption, or convergence
claim. R010 supersedes the experiment it previously motivated.

Raw logs remain under `logs/` and code-local `run_logs/`. This file stores only evidence needed for research decisions.

## E014 - Targeted multi-system dynamics literature study

Date: 2026-09-05 UTC. Research evidence only, not a graph training experiment.

Read methods of Hamiltonian meta-learning (ICLR 2021, cross-parameter;
ICLR 2024, cross-system-type few-shot adaptation), MP-NODE (NeurIPS 2022,
homogeneous coupled modules), and NCF (ICLR 2025, context self-modulated fields).
Compared CoDA, GG-ODE, GREAT and LEADS. Primary-source links, assumption
boundaries and project-specific inferences are in
`.collab/MULTISYSTEM_DYNAMICS_LITERATURE.md`.

Structured generators and learned dynamical interfaces are useful references;
none validates the current autonomous proxy or directly solves private graph
knowledge exchange. Keep backbone choice open and require independent task
evidence beyond fitting self-generated trajectories. R008 NEEDS_RESEARCH.

## E013 - CPU audit: ridge equivalence, native causal response and realization

Date: 2026-09-05 UTC. Seed 20260905, CPU float64, one thread. Complete diagnostic
and reproduction command: `.collab/DYNAMICAL_RESPONSE_CHECKS.md`.

| Check | Result |
| --- | --- |
| Actual A ridge versus matched P ridge | `||P-I-hA||_F = 2.162e-16` |
| Native 16-step forward versus composed native steps | maximum error `0` |
| Native causal kernel AD versus central difference | relative error `4.025e-11` |
| Future intervention affecting earlier outputs | maximum response `0` |
| Perturbation remainder as radius halves | approximately fourfold reduction |
| Separate toy feasible conservative flow | linear conservation residual `6.360e-17` |
| Toy nonlinear finite update | quadratic conservation drift; four tested energy changes negative |
| Orthogonal reachable response directions | zero conservative update despite nonzero disagreement |

Changing A to identity-centered P is not an expressivity fix. Exact discrete
response is measurable without the old proxy on this random-initialized
12-node model. Local feasibility changes conservative-flow equilibria and
finite realization is not exactly conservative. No task benefit, trained-model
coverage, privacy or training-cost conclusion follows. The flow toy is not the
graph response model. No production source, checkpoints or training changed.

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

## E008 - Method v0 conservative dynamical knowledge-flow specification

### Scope

Research specification only; no source modification, new training run, or new
test was performed in this entry.

### Result

`methodv0.md` restores conservative dynamical knowledge flow as the single
paper-level story. It defines a client dynamical state, a task-relevant
response observation after cross-graph transport, a frozen-exchange conserved
weighted first moment, a response heterogeneity energy, a conservative flux,
local admissibility, and a realization residual linking public flow back to the
local graph system.

The ideal fixed-coordinate flow satisfies

\[
\sum_m p_mJ_m=0,
\qquad
\frac{d\mathcal E_{\mathrm{het}}}{d\tau}
=-2\kappa\mathcal E_{\mathrm{het}}.
\]

These equations are an analytical baseline, not evidence of task improvement
or a replacement name for ordinary consensus. The implementation claim is
conditional on a non-degenerate task-relevant observation operator, measurable
transport/realization error, and local admissibility constraints.

### Next validation question

Choose one concrete observation operator, admissibility approximation, and
local realization method. Then test whether the complete loop produces
conserved exchange, reduced task-relevant response energy, bounded realization
error, and task behavior consistent with the claimed mechanism.

## E009 - Method v0 toy conservative exchange

### Configuration

- 10 clients, 6-dimensional fixed transported dynamical states, equal weights;
- \(\delta\tau=0.25\), \(\kappa=0.4\), 20 exchange steps;
- one fully connected interaction graph and one graph with two admissible
  client groups; symmetric edge gates;
- no graph Laplacian construction, SVD, model training, or node-level state.

### Results

| Interaction graph | Initial energy | Final energy | Final/initial | Monotone steps | Max conservation residual |
| --- | ---: | ---: | ---: | --- | ---: |
| Fully connected | 4.69634122 | 0.06941607 | 0.01478088 | 20/20 | `4.25e-17` |
| Two admissible groups | 4.69634122 | 3.28964478 | 0.70046971 | 20/20 | `7.49e-17` |

The fully connected ratio matches \((1-\kappa\delta\tau)^{40}=0.9^{40}\).
The gated case keeps a nonzero cross-group residual while decreasing energy at
every step. This is an initial mechanism result for the exchange layer only;
it does not establish graph task improvement or validate the local realization.

### Decision use

The result supports advancing Method v0 to a minimal opt-in implementation.
The first implementation experiment should test whether the same conservation,
energy, and non-consensus behavior survives after trajectory measurement,
Functional Map transport, local low-rank realization, and local task training.

## E010 - Low-dimensional generator fit is the current bottleneck

### Observation

The current ridge generator fits one-step velocities on the calibration window.
The 11-dataset steady-state median errors range from `0.276` to `0.727`.
Generator fit quality does not visibly align with the recorded task gains, and
the current logs do not contain held-out time-block or multi-step rollout
errors.

### Interpretation

These values are too large to treat the generator as validated transferable
dynamical knowledge. They may reflect nonlinear vector fields, basis projection
loss, finite-difference noise, hidden time variation, or a mismatch between an
autonomous linear generator and the actual Euler transition. The exchange-layer
result E009 therefore cannot be combined with current generator logs to claim a
graph task result.

### Research action

Compare the current continuous generator against a directly fitted discrete
propagator and an affine propagator using convex ridge objectives. Evaluate on
held-out time blocks, multi-step rollouts, trajectory statistics, transport
residuals, and local task behavior. A small neural residual is a later
nonconvex diagnostic only if the convex candidates fail.

### Status

Implementation is paused at R007 until probe response transport and realization
pass the validation gate.

## E011 - Public state redefined as common-probe finite-time response

### Research decision

Following the generator-fit diagnosis, Method v0 no longer uses the artificial
state `vec(z_m^0) || vec(A_m)` as the primary exchanged knowledge. A fixed public
probe bank is transported into each client, propagated by the actual finite
time graph dynamics, transported back, and vectorized across probe/time
responses. This response state directly represents the behavior of each graph
system under the same stimulus.

### Consequence

The current generator error becomes a compression diagnostic. A large error no
longer invalidates direct response exchange, but it blocks replacing raw
responses with a generator unless held-out probe rollout passes. Local
realization is measured by a response Jacobian and a constrained convex solve,
then checked with an actual post-update rollout.

### Status

Conceptual update only; no source or new graph training run. R007 remains
`NEEDS_RESEARCH` until probe response transport and realization are validated.

## E012 - Causal response-kernel candidate

### Research update

The public-probe response idea is refined into a causal kernel: inject a
specified dynamical perturbation through an explicit port at time `s`, observe
the task-relevant response at time `t`, and retain the port, `s`, `t`, output
semantics, and sign. The kernel is obtained from the actual ODE/Euler
variational dynamics, rather than from an autonomous low-dimensional generator.

### Why this addresses E010

The generator's `0.276-0.727` in-window residual no longer invalidates the
knowledge object, because the kernel directly measures finite-time response.
The generator, discrete operator, and neural model remain optional compression
comparisons and must pass held-out kernel-response validation.

### Next check

On a frozen synthetic graph/model, verify zero-port equivalence, future-to-past
causality, float64 AD versus centered finite differences, nonlinear remainder
versus perturbation amplitude, response coverage, and finite-step realization
drift. No large graph Laplacian SVD or node-level SVD is permitted.

### Status

Conceptual candidate only; no source or new graph training run. R008 remains
`NEEDS_RESEARCH`.

## E004 - Design diagnosis: small effect is structural, not only a prototype issue

### Observation

The current correction is a one-step residual around a server prototype. Its
steady-state magnitude is only 0.57%-1.32% of the native field, and the
dissipative projection acts on most datasets. Calibration then turns the
correction off, so the next native generator is not the generator of the
controlled system. The implementation is stable, but there is no explicit
round-to-round objective that minimizes cross-client dynamical dispersion.

### Interpretation

The current design can be viewed as safe transfer rather than dynamical
consensus. A zero prototype-switch count is a secondary symptom: changing the
prototype would not by itself create a contraction objective. The shape/speed
split also makes the shared target intentionally partial, which is reasonable
for task preservation but cannot support a claim of minimizing full dynamical
heterogeneity.

### New research direction

Redesign around a measurable heterogeneity energy over effective initial-state
and vector-field summaries. Use a weighted canonical barycenter and a native
proximal anchor, so the client update is an explicit contraction toward the
shared target while retaining a bounded client-native residual. Enforce the
stability condition on the total effective generator/Euler map. Prototype
clustering should remain secondary until this contraction claim is tested.

### Status

Research hypothesis only. No source implementation or prototype-control runs
are authorized by the current handoff until the energy, feedback law,
non-circular timing, and stability surrogate are fully specified.

## E005 - Review of the conservative dynamical knowledge-flow skeleton

### Assessment

The new high-level document provides a stronger research motivation than the
current prototype residual: it introduces two time scales, transport before
exchange, an explicit heterogeneity potential, conservative inter-client flux,
and an admissible minimum-heterogeneity equilibrium. These are appropriate
objects for a theory-to-algorithm program.

### Necessary refinement

The condition `sum_m p_m J_m = 0` is conservation of a canonical exchange flux,
not by itself conservation of dynamical knowledge. A variance energy over raw
`Xi_m` can be minimized by destructive homogenization and is not yet a measure
of harmful heterogeneity. The final energy should compare transported
vector-field/flow responses on shared semantic probe states and include a
task-preservation or native-residual constraint. Coordinate mismatch must be
removed before measuring intrinsic disagreement.

### Candidate insight

Define the transferable component as the part of client dynamics that produces
different responses to the same transported semantic probes. Let the remaining
client-native component be an admissible residual. Use a task-aware constrained
gradient flow, or pairwise antisymmetric flux with weighted detailed balance,
to dissipate only the transferable disagreement. This yields a meaningful
claim: the harmful component contracts, while intrinsic client individuality is
not forced to zero.

### Status

High-level skeleton accepted as the current research direction. The knowledge
object, energy, flux, stability condition, and non-circular timing remain open;
no source implementation should begin yet.

## E006 - Dissipative correction is not a required high-level principle

The original A-DGN anti-symmetric/dissipative parameterization belongs to the
backbone's deep continuous-depth stability motivation. The proposed federated
knowledge-flow problem has a different requirement: reduce a task-aware
heterogeneity energy over exchange time \(\tau\). These are separate axes.

The current `minimal_dissipative_projection` enforces a negative-semidefinite
symmetric part on the correction itself, then clips its spectral norm. This
can remove the positive symmetric component that would move a client toward a
consensus target, while it still does not prove stability of the nonlinear
total field. Finite-horizon numerical and gradient behavior should be assessed
for the actual chosen field. A low-rank fitted \(\|I+\Delta t A_m^{eff}\|\)
would be only a surrogate, not a certificate for the nonlinear rollout; no
trust-region or backtracking construction is mandatory at this stage.

Therefore the next high-level design should retain `heterogeneity dissipation`
as the objective and treat ODE dissipativity as optional implementation
regularization. Removing or weakening correction dissipativity is a research
hypothesis that must be compared against the current projection, not an
immediate source edit.

## E007 - General native weights and a correction-projection counterexample

Date: 2026-09-05 UTC. Code inspection and analytic example only; no experiment
or test was run for this finding.

The active native field uses the upstream \(W-W^\top-\gamma I\), while
the correction separately has positive symmetric eigenvalues zeroed before
spectral clipping. These are distinct constraints. The first is an optional
backbone inductive bias for the proposed research; the second can exclude a
useful correction even when the resulting system remains stable.

For a scalar native field \(\dot h=-2h\) and target \(\dot h=-h\), the
required correction is \(+h\). The code's dissipative projection maps this
scalar correction to zero. Applying the unprojected correction instead gives
the stable target system. At Euler step size 0.1 both original and target maps
have factors 0.8 and 0.9, respectively, so the example also holds discretely.
It establishes that correction dissipativity is not necessary for stable
alignment, not that removing projection will improve the recorded datasets.

The observed small injection cannot be causally attributed to positive
eigenvalue removal from the available aggregate diagnostics. Beta, subsequent
spectral clipping, subspace coverage, and action on visited states also matter.
The proposed high-level flow does not require antisymmetric native weights.

Evidence: the upstream graph_heteropily/models/antisymmetric_dgn.py, and the
active backbone_functional_dynamics_stable/models/s0/model.py and
models/dissipative_injector.py.

Scope correction for E004/E005: native calibration still receives parameter
feedback from injected training; the old ideal unprojected residual already
interpolates generators. Proximal barycenters and shared semantic probes remain
candidates. None of the existing diagnostics proves round-to-round contraction
or explains the small accuracy changes causally.
