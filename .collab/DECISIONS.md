# Design Decisions

Updated: 2026-09-06 UTC

## D020 - Define equilibrium by exhaustion of external learnable gain

Decision:

Adopt finite complementary knowledge absorption as the learning-process premise
for the next method revision. For client `i`, define federated value as the
incremental task gain of receiver-realizable external dynamical-response
directions over a compute- and norm-matched local-only update. Cross-client flow
is active only while this gain is positive. The equilibrium criterion is that
no client has a significant positive external marginal gain.

Response heterogeneity remains a diagnostic and a source of candidate
directions, but minimizing it is not sufficient: large response differences may
be useless or infeasible, and useful external knowledge may remain even when a
global distance is small. The antisymmetric conservation law is demoted from a
learning axiom to an optional centering/numerical constraint because knowledge
is non-rival and is not depleted at the sender when learned by a receiver.

Scope:

The finite knowledge-pool statement is a bounded, stage-wise assumption under
a fixed task, client population, model class, and communication interface. The
pool may refresh as clients learn, so marginal gain is re-estimated each round.
This decision does not establish task benefit or convergence in the current
nonconvex implementation.

Status: Active research direction; R009 `NEEDS_RESEARCH`.

## D019 - Consolidate the canonical method specification in Method v0

Decision:

Use the front-loaded agent-transfer section of `methodv0.md` as the canonical
method specification. It absorbs the later causal response-kernel definition,
the correction from sequential summary-flow realization to a joint realizable
conservative solve, the A/P ridge equivalence, lifecycle placement, and the
current validation gates. `.collab/NEXT_TASK.md` remains the canonical
operational handoff, while the longer `.collab` research notes provide evidence
and derivations.

Reason:

The previous state split the latest method across `methodv0.md`, a newer kernel
note, decisions, and the handoff. A new agent could mistakenly implement the
older random-probe/sequential-flow version or treat an open backbone candidate
as adopted. The consolidated header identifies fixed choices, open choices,
and the only authorized next research step.

Status: Active documentation decision. Research remains `NEEDS_RESEARCH`; no
production implementation or training is authorized.

## D018 - Prioritize structured dynamics and learned interaction interfaces

The user prioritizes Hamiltonian meta-learning and NeurIPS 2022 MP-NODE.
Primary-source comparison: `.collab/MULTISYSTEM_DYNAMICS_LITERATURE.md`.
Communication must describe dynamical action or response, distinguish
propagation time from exchange time, and be validated with interventions plus
task evidence. Hamiltonian functions are not scalar energy summaries;
MP-NODE messages are not proven conserved physical fluxes. NCF (ICLR 2025)
provides a newer reference for cross-environment nonlinear field learning.
None automatically supplies a private heterogeneous-graph FL protocol.

The response kernel is a candidate interface, not an adopted final backbone.
Compare native response exchange, context-conditioned module fields and
augmented message dynamics according to the failure each addresses. Do not
automatically stack them, impose canonical Hamiltonian coordinates, or force
a shared/private decomposition. Physical-trajectory supervision in the papers
does not validate task information in our self-generated hidden trajectories.

Status: Active research priorities; R008 NEEDS_RESEARCH. No source change.

## D017 - Continuous/discrete ridge equivalence corrects the earlier comparison

With matched data, basis, weights and regularization, substituting `P=I+h*A`
in `||Zplus-PZ||^2 + lambda*||P-I||^2` yields `h^2` times
`||(Zplus-Z)/h-AZ||^2 + lambda*||A||^2`. The optimum is identical, including
the implementation's trace-scaled ridge. Different relative-error denominators
must not be used as evidence of better identification.

This supersedes the expectation in D014/E010 and Method v0 section 10.8 that
changing to this P objective alone fixes model expressivity or statistical
noise. Historical records and v0 are preserved; future work must use this
correction. Affine, memory or nonlinear alternatives can change the model class
and remain unvalidated.

E013 records native derivative checks and a separate toy conservative-flow
calculation. Their scope is numerical consistency, not response sufficiency,
task transfer, or new-backbone efficacy.

Status: Algebraic correction adopted; research candidates not yet adopted.

## D016 - Upgrade the candidate knowledge object to a causal response kernel

Decision:

The common-probe idea is refined into a candidate finite-time causal response
kernel. A client receives a defined dynamical port perturbation at time `s`,
propagates the actual graph ODE/Euler dynamics, and reports how a fixed task
observation changes at time `t`. The public object retains port type, `s`, `t`,
output semantics, and response sign. It is not a flattened initial-state and
generator summary.

The kernel is computed from the true variational dynamics or exact discrete
JVPs. Parameter Jacobians are used only to find locally realizable updates;
they are not the exchanged knowledge. A low-dimensional generator remains a
compression diagnostic and must pass held-out response validation before use.

The first validation is on a frozen synthetic model: zero-port equivalence,
causality, AD/finite-difference agreement, nonlinear remainder versus probe
amplitude, response coverage, and finite-step conservation drift. Large graph
Laplacian eigenvectors and SVD are outside the method.

Reason:

The causal kernel gives the exchanged object a direct dynamical meaning: it
measures how the same intervention is transmitted through different graph
systems. It avoids claiming that a poorly fitted autonomous low-dimensional
operator represents the full nonlinear dynamics.

Status: Active research candidate; no source implementation authorized yet.

## D015 - Define public knowledge by common-probe finite-time response

Decision:

Method v0 does not define the exchanged state by concatenating
`vec(z_m^0)` and `vec(A_m)`. That construction is too dependent on arbitrary
summary choices and can reduce the method to ordinary consensus on a feature
vector. The public state is instead the transported finite-time response to a
fixed public probe bank:

\[
x_m=\operatorname{vec}\left[
\mathcal T_m\mathcal O_m
\left(\Phi_{m,\ell}(\mathcal T_m^{-1}P_q^c)
 -\Phi_{m,0}(\mathcal T_m^{-1}P_q^c)\right)
\right]_{q,\ell}.
\]

The same probe semantics and time points are used for every client. The
heterogeneity energy therefore measures disagreement in actual propagation
responses of the graph dynamical systems. A low-dimensional generator is only
a compression/diagnostic model; it is not accepted as knowledge unless it
reproduces held-out probe responses.

Local realization is defined by the Jacobian of the probe-response map with
respect to a low-rank field correction, followed by a constrained convex
quadratic solve and an actual post-update rollout. Raw response exchange is
allowed in the first experiment if it is small enough; compression is not
forced prematurely.

Reason:

The method's novelty depends on exchanging dynamical behavior under the same
stimulus, not on averaging arbitrary coordinates. This also makes a large
generator-fit error diagnosable rather than silently conserved.

Status: Active research direction; R007 remains research-stage.

## D014 - Treat low-dimensional operator identification as a prerequisite

Decision:

Do not treat the current ridge-fitted continuous generator as valid dynamical
knowledge by default. Its logged relative in-window velocity residual is about
`0.276-0.727` at steady state across the current 11-dataset matrix, and it has
not been tested on held-out time blocks or multi-step rollout. Before flow
implementation, compare the continuous generator with a directly fitted
discrete propagator and an optional affine propagator using convex ridge
objectives.

The first candidate is

\[
\min_P \sum_t\|z_{t+1}-Pz_t\|^2+\lambda\|P-I\|_F^2,
\]

because it fits the actual finite-step Euler transition without amplifying
difference noise by `1/delta_t`. A small neural residual is a later diagnostic,
not the default: its fitting objective is nonconvex and its parameters are not
automatically transportable or compatible with the conservation argument.

Use time-block validation, multi-step rollout error, trajectory-response error,
transport error, and local task behavior as the acceptance gate. Initial
experimental thresholds are one-step `0.15`, multi-step `0.25`, and realization
residual `0.25`; these are tunable reported hyperparameters, not theoretical
constants.

Numerical policy remains strict: no SVD on large graph Laplacians, adjacency
matrices, or node-level feature matrices. QR and ridge solves are sufficient
for the first operator comparison.

Reason:

The conservative flow can only carry knowledge that is actually identified.
A beautiful conservation law applied to a poor surrogate would conserve model
error rather than useful graph dynamics.

Status: Active research direction; source implementation paused pending the
operator validation gate.

## D013 - v0 uses convex gated edge flow and conservative numerical linear algebra

Decision:

The first implementation of Method v0 uses one public transported response
state, a fixed-gate convex quadratic exchange problem, and pairwise symmetric
edge flows. FedAvg sample weighting remains a baseline parameter aggregation
choice; it is not automatically used as the dynamical knowledge metric. The
first toy result uses uniform knowledge weights, with sample-size and
confidence-based alternatives treated as explicit controls.

The numerical policy is strict: no SVD is allowed on a large graph Laplacian,
adjacency matrix, or node-level feature matrix. The v0 trajectory basis uses
deterministic QR, generator fitting uses ridge `solve`, and any SVD retained in
legacy low-dimensional Functional Map utilities must be dimension-capped and
reported as a diagnostic or replaced before scaling.

Reason:

The convex edge-flow objective makes the exchange target a learned constrained
optimization problem rather than an unexplained weighted average. Symmetric
gates preserve the conservation proof under local admissibility. The toy
experiment gives initial evidence: full exchange reduced energy to 1.478% of
its initial value, while two admissible groups retained 70.047% with monotone
descent; maximum conservation residual was below `8e-17`.

Status: Active; ready for minimal opt-in implementation.

## D012 - Method v0 adopts conservative dynamical knowledge flow as the single high-level story

Decision:

Adopt `methodv0.md` as the current research specification. The paper-level
object is a task-relevant dynamical response state transported into a common
functional coordinate. The mechanism is a conservative client-to-client
knowledge flux driven by an explicitly defined heterogeneity energy. The flux
must be realized in each local graph dynamical system and evaluated under local
task and dynamical admissibility constraints.

The central chain is:

`heterogeneous graph dynamical systems -> knowledge transport -> conservative
knowledge flow -> heterogeneity-energy dissipation -> admissible dynamical
equilibrium`.

The weighted first moment is conserved during a frozen exchange phase only.
Local learning is a separate source term. The unconstrained mean-attraction
flow is retained as the analytical baseline for conservation and dissipation;
it is not itself the claimed algorithm. The final equilibrium need not be full
consensus because local admissibility can block harmful homogenization.

Proximal consensus, multi-prototype routing, correction-only dissipativity,
and native anti-symmetry are implementation candidates or controls, not the
method's motivation or universal requirements.

Reason:

This gives the work a causal object and mechanism before choosing engineering
tools. It also makes the main claim falsifiable through conservation residual,
energy change, realization error, and task behavior.

Status: Active research direction; concrete observation and realization remain
to be selected and validated.

## D001 - Fixed collaboration roles

Decision:

- `root` is the research agent responsible for idea improvement, theory, diagnosis, experiment interpretation, and implementation-ready specifications.
- `xzc` is the implementation agent responsible for code, debugging, tests, and experiments.

Reason:

The two accounts do not share chat context. Fixed ownership and file-based handoffs prevent the human from becoming a message relay.

Status: Active.

## D008 - Candidate direction: proximal dynamical consensus

Scope correction (2026-09-05 UTC): this is an illustrative historical candidate,
not an adopted architecture. The old unprojected residual already interpolates
toward a target in an ideal fixed-coordinate model. Renaming that step proximal
consensus is not sufficient novelty. Its algebraic contraction does not prove
contraction across nonlinear federated training rounds. Mandatory backtracking
below is superseded by D010/D011.

Decision:

The leading redesign candidate is a proximal consensus controller over
trajectory-induced low-dimensional summaries. It is a research hypothesis,
not yet an implementation handoff.

For each client, represent the native summary in canonical coordinates as

\[
s_m=(z_m^0,\;\widehat A_m,\;\ell_m),
\qquad
\ell_m=\log(\|A_m\|_F+\epsilon),
\]

where (z_m^0) is an initial-state descriptor, \(\widehat A_m\) is generator
shape, and \(\ell_m\) retains propagation speed. The server computes a weighted
barycenter (s_*\), and each client solves a proximal step toward that target:

\[
s_m^{eff}
=
\arg\min_s
\frac{\kappa}{2}\|s-s_*\|^2
+\frac{\lambda_m}{2}\|s-s_m^{native}\|^2.
\]

Thus (s_m^{eff}=s_m^{native}+\eta_m(s_*-s_m^{native})), with
\(\eta_m=\kappa/(\kappa+\lambda_m)\). For a fixed target and common \(\eta\),
the weighted disagreement energy contracts by \((1-\eta)^2\). Target drift,
map error, and generator-fit error become explicit additive terms in the
empirical contraction diagnostic.

The controller must backtrack the coupling strength against stability of the
**total** effective generator (A_m^{native}+U_m). Projecting only (U_m) is
not sufficient because it can delete the consensus direction itself.

Status: Proposed; requires theoretical and synthetic validation before source
implementation.

## D009 - High-level skeleton accepted; task-aware conservative flow required

Decision:

`CONSERVATIVE_DYNAMICAL_KNOWLEDGE_FLOW_HIGH_LEVEL.md` is accepted as the
research-level conceptual skeleton, not as an implementation specification.
The central claim is refined as follows: federated exchange should dissipate
only transferable, task-harmful dynamical disagreement while preserving locally
admissible task behavior. Shared semantic probes and explicit native residuals
are candidates, not mandatory choices in the high-level skeleton.

The next theory must distinguish conservation of an exchange flux from
conservation of a physical quantity, coordinate/gauge mismatch from intrinsic
dynamical heterogeneity, harmful disagreement from task-required individuality,
and summary variance from effective-trajectory improvement.

The preferred direction is a task-aware constrained gradient flow or pairwise
antisymmetric flux in a transported knowledge space. A simple weighted mean of
raw generators is insufficient as the final motivation.

Status: Active research direction; no source implementation authorized.

## D010 - Separate heterogeneity dissipation from ODE dissipativity

Decision:

Do not treat dissipativity of the injected correction as a fundamental
requirement of the new method. The fundamental requirement is descent of a
task-aware heterogeneity energy along federated exchange time \(\tau\). Local
representation propagation along \(t\) has a separate finite-horizon numerical
and optimization question; no particular stability constraint is mandated yet.

The existing A-DGN anti-symmetric parameterization remains a property of the
baseline backbone. It is not a reason to force every federated correction to
have a negative-semidefinite symmetric part. A total-field trust region or
backtracking could be considered if the selected field requires it, but is not
a replacement architectural requirement. A bound on a fitted low-rank
\(\|I+\Delta t A_m^{eff}\|\) is only a surrogate: it does not certify the full
nonlinear Euler rollout or its gradients.

Reason:

The current `minimal_dissipative_projection` acts on the correction alone and
can delete the direction needed to reduce client disagreement. It also does
not establish stability of the nonlinear graph field or of the total effective
Euler update. Since the project uses a finite fixed number of steps, the
asymptotic depth-stability motivation of the original A-DGN paper is stronger
than required for the proposed federated knowledge-flow objective.

Status: Active research direction; source implementation remains paused.

## D011 - Treat A-DGN antisymmetry as a baseline choice, not a requirement

Decision:

The ODE viewpoint and the federated heterogeneity objective do not require
the native matrix to be parameterized as \(W-W^\top-\gamma I\), nor do they
require the injected correction to be negative-semidefinite. A finite Euler
rollout can use a general graph vector field, for example
\[
\dot H=\sigma(HW+\operatorname{GConv}(H,G)+b),
\]
without requiring an asymptotic contraction theorem for local propagation.
Finite-horizon numerical behavior and gradient behavior still need assessment.

The official anti-symmetric A-DGN remains the comparison baseline. Any simpler
native field is a separate backbone variant and must be evaluated as such; it
must not be presented as an unchanged A-DGN result. The proposed contribution
should be motivated by task-aware contraction of transferable dynamical
disagreement, independently of this backbone parameterization.

Reason:

The anti-symmetric self-interaction is norm-preserving in its isolated linear
continuous-time form; the negative diagonal adds damping. These properties of
one component do not by themselves certify the graph-coupled nonlinear field
or its explicit Euler discretization. A finite rollout weakens the motivation
for imposing deep-propagation constraints as universal axioms. A general matrix
permits more symmetric directions, but an accuracy benefit is unproven.
Anti-symmetrization itself is cheap and uses the existing dense parameter:
removing it changes inductive bias rather than materially reducing parameter
count. No new trust-region or backtracking architecture is selected.

Status: Active research direction; no source change authorized.

## D007 - Redesign around heterogeneity contraction before prototypes

Decision:

Pause prototype-specific engineering. Redesign the method as a coupled
dynamical-system consensus problem whose explicit objective is to reduce the
dispersion of the clients' **effective** dynamics while keeping a proximal
penalty toward each native client system.

The next design must define a measurable heterogeneity energy over both
initial-state and vector-field components, derive a client feedback law that
contracts that energy, and enforce stability on the total effective field. A
prototype bank is not the primary mechanism and must not be expanded until
this contraction claim is validated.

Reason:

The current path minimizes neither a global dispersion objective nor a
round-to-round contraction bound. It applies a small residual correction to a
native trajectory, calibrates the next generator with correction disabled, and
projects the correction itself to be dissipative. These choices make the
intervention safe but do not imply that the resulting client systems become
more similar. The shape/speed split also permanently excludes speed from the
consensus target, so full dynamical heterogeneity cannot converge to a minimum
under the current objective.

Status: Historical redesign proposal; mandatory proximal/consensus and stability
choices are superseded by the open skeleton in D009-D011 and R003.

## D002 - Repository state is the shared memory

Decision:

Use `AGENTS.md` plus `.collab/` as the durable protocol. Keep only current state, decisions, tasks, and research-relevant evidence; do not store full chat transcripts.

Reason:

These files are versionable, searchable, and recoverable by either account in a fresh session.

Status: Active.

## D003 - Preserve the official training and evaluation backbone

Decision:

Research additions remain opt-in and preserve official A-DGN, the FedAvg lifecycle, persistent optimizer state, and per-client best-validation paired test evaluation unless a later handoff explicitly changes an invariant.

Reason:

This keeps comparisons attributable to the research mechanism.

Status: Active.

## D004 - Functional Dynamics remains experimental

Decision:

Treat the current multi-prototype Functional Dynamics implementation as a candidate mechanism rather than an accepted final method.

Reason:

The seed-42 11-dataset matrix shows modest, nonuniform changes and lacks the matched controls needed to establish mechanism validity. Diagnostics also show zero prototype switches under margin 1.0, small effective injection, and frequent spectral clipping.

Status: Superseded by redesign handoff R003.

## D005 - Start with a file protocol, without automatic watchers

Decision:

Use manual account startup plus automatic reading of repository instructions. Do not add an `inotify`/`codex exec` relay during the first phase.

Reason:

The handoff format and concurrent-edit discipline should stabilize before unattended execution is introduced.

Status: Active.

## D006 - Controls before increasing transfer strength

Decision:

Do not change the default injection coefficient, spectral cap, or prototype
switch rule yet. First implement and run matched controls that separate
cross-client transfer from the computation path, fixed clustering, and a
wrong prototype.

Reason:

The current correction is deliberately weak: the steady-state injected field
is about 0.57%-1.32% of the native field, and the spectral cap removes a
substantial fraction of the raw correction on most datasets. Prototype
assignments also never switch under the current margin. Increasing the signal
before a causal control would confound useful transfer with generic stronger
regularization or instability.

Adopted controls:

- module-off: exact official A-DGN + FedAvg baseline;
- local-only/no-transfer: functional-dynamics path runs, but the client uses
  its own local summary as target, producing zero cross-client correction;
- single-prototype: one canonical prototype;
- wrong-prototype: deterministic incompatible prototype selected with the same
  map solver and safety projection;
- full: current multi-prototype method.

Status: Superseded by D007/R003; retain as a later ablation plan.
