# Conservative Dynamical Knowledge Flow for Heterogeneous Graph Federated Learning
## High-Level Research Skeleton

> **Scope of this document**
> This file only fixes the **high-level research idea and problem formulation**.
> Any concrete implementation mentioned below—such as Functional Map, low-dimensional generator fitting, trajectory basis, barycenter computation, dissipative projection, prototype learning, or specific ODE solvers—should be treated only as **candidate technical tools**.
> They are **not part of the final method definition yet** and remain open for further exploration.

---

# 1. Core Problem

In heterogeneous graph federated learning, different clients possess different graph structures, feature distributions, label compositions, and local propagation environments.

Instead of viewing such heterogeneity only as parameter inconsistency, we model each client as a private graph dynamical system:

$$
\mathcal S_m
=
\left(
H_m^0,
F_m
\right),
$$

with representation evolution:

$$
\frac{dH_m(t)}{dt}
=
F_m(H_m(t)).
$$

Here:

- $H_m^0$ represents the client-specific initial state;
- $F_m$ represents the client-specific graph propagation dynamics;
- feature heterogeneity mainly changes the initial condition;
- structural heterogeneity changes the vector field / propagation operator;
- their joint effect appears as heterogeneous representation trajectories.

Therefore, the fundamental federated problem is not simply:

$$
\text{how to average heterogeneous model parameters},
$$

but rather:

$$
\boxed{
\text{how heterogeneous graph dynamical systems can exchange useful dynamical knowledge}
}
$$

while preserving client-specific admissible dynamics.

---

# 2. High-Level Insight

The central view is:

> **Each client carries a portion of dynamical knowledge. Federation should allow this knowledge to flow across heterogeneous graph systems in a conservative and constrained manner, so that harmful dynamical heterogeneity is gradually dissipated while useful local individuality is preserved.**

This leads to the main conceptual skeleton:

$$
\boxed{
\text{Heterogeneous Dynamical Systems}
\rightarrow
\text{Conservative Knowledge Flow}
\rightarrow
\text{Heterogeneity Dissipation}
\rightarrow
\text{Minimum-Heterogeneity Dynamical Equilibrium}
}
$$

This is the central storyline of the project.

The method should not be presented as:

- another global/local model decomposition;
- another personalized aggregation rule;
- another prototype clustering method;
- another gradient conflict correction;
- another heuristic residual injection.

Instead, the method should be presented as:

$$
\boxed{
\text{federated learning as conservative dynamical knowledge redistribution}
}
$$

---

# 3. What Is “Dynamical Knowledge”?

Each client contains information not only about its current representation state, but also about how that state evolves.

Abstractly, define a client-level dynamical knowledge state:

$$
\Xi_m.
$$

The exact mathematical form of $\Xi_m$ is **not fixed yet**.

It may eventually contain some combination of:

$$
\Xi_m
=
\left(
\text{initial-state knowledge},
\text{vector-field knowledge},
\text{finite-time flow knowledge},
\text{trajectory statistics},
\ldots
\right).
$$

The only requirement at the high level is:

> $\Xi_m$ should summarize the aspects of the local graph dynamical system that are relevant for cross-client collaboration.

Possible realizations include, but are not limited to:

- reduced generators;
- finite-time propagators;
- trajectory operators;
- Koopman-style observables;
- functional coordinates;
- trajectory moments;
- task-aware dynamical summaries.

These are implementation candidates, not fixed design choices.

---

# 4. Why Knowledge Must Be Transported Before It Can Flow

Different clients live on different graphs, with different node sets and different local representation spaces.

Therefore, even if two clients both possess dynamical knowledge:

$$
\Xi_m,
\qquad
\Xi_n,
$$

they generally cannot be compared or directly combined in their original coordinates.

Hence, before knowledge can flow, a cross-system transport mechanism is required:

$$
\mathcal T_m:
\mathcal F_m
\rightarrow
\mathcal F_c,
$$

where $\mathcal F_c$ is some shared or canonical functional space.

After transport:

$$
\widetilde\Xi_m
=
\mathcal T_m(\Xi_m).
$$

The purpose of $\mathcal T_m$ is not merely representation alignment.

Its conceptual role is:

$$
\boxed{
\text{make dynamical knowledge transportable across heterogeneous graph systems}
}
$$

A Functional Map is currently one promising candidate for realizing $\mathcal T_m$, because it naturally maps functions/operators across different spaces.

However:

> **Functional Map is not fixed as the final implementation.**

Other transport mechanisms are still admissible if they better satisfy scalability, stability, privacy, and dynamical consistency.

---

# 5. Conservative Dynamical Knowledge Flow

Once local dynamical knowledge has been transported into a common comparable space:

$$
\widetilde\Xi_1,
\ldots,
\widetilde\Xi_M,
$$

we introduce a federated knowledge flow.

Let:

$$
J_m
$$

denote the net dynamical knowledge flux entering client $m$ in the canonical space.

The first fundamental requirement is **conservation**:

$$
\boxed{
\sum_m p_mJ_m
=
0.
}
$$

Therefore:

$$
\frac{d}{d\tau}
\sum_m
p_m\widetilde\Xi_m
=
0.
$$

Here $\tau$ is the federated interaction time, distinct from the local graph propagation time $t$.

The interpretation is:

> federation does not arbitrarily create or destroy global dynamical knowledge; it redistributes knowledge among clients.

This is the first key structural property of the method.

---

# 6. Heterogeneity as an Energy / Potential

To drive the knowledge flow, define a global dynamical heterogeneity energy:

$$
\mathcal E_{\mathrm{het}}
=
\mathcal E
\left(
\widetilde\Xi_1,
\ldots,
\widetilde\Xi_M
\right).
$$

The exact form of $\mathcal E_{\mathrm{het}}$ is also **not fixed yet**.

A simple conceptual example is:

$$
\mathcal E_{\mathrm{het}}
=
\frac12
\sum_m
p_m
\|
\widetilde\Xi_m-\bar\Xi
\|^2,
$$

where:

$$
\bar\Xi
=
\sum_m
p_m\widetilde\Xi_m.
$$

But the final heterogeneity energy should likely distinguish:

- coordinate mismatch;
- truly harmful dynamical disagreement;
- task-relevant local individuality;
- non-transferable client-specific dynamics.

Therefore, the final paper should not simply equate:

$$
\text{heterogeneity}
=
\text{distance to global mean}.
$$

Instead, the research goal is to identify a physically and task-consistent energy whose decrease corresponds to **reduced harmful dynamical heterogeneity**.

---

# 7. Knowledge Flow Should Dissipate Heterogeneity

The second fundamental requirement is:

$$
\boxed{
\frac{d\mathcal E_{\mathrm{het}}}{d\tau}
\le
0.
}
$$

That is, the federated knowledge flow should continuously dissipate cross-client dynamical heterogeneity.

The simplest idealized flow is:

$$
\frac{d\widetilde\Xi_m}{d\tau}
=
\kappa
\left(
\bar\Xi-\widetilde\Xi_m
\right),
$$

which simultaneously satisfies:

$$
\sum_m
p_m
\frac{d\widetilde\Xi_m}{d\tau}
=
0
$$

and:

$$
\frac{d\mathcal E_{\mathrm{het}}}{d\tau}
\le
0.
$$

This simple form is only an illustrative prototype.

The final method may use:

- pairwise flux;
- graph-of-clients diffusion;
- constrained gradient flow;
- optimal transport;
- Wasserstein / Bregman flow;
- operator-valued diffusion;
- finite-time conservative transport;
- other structure-preserving fluxes.

The high-level requirement is more important than the specific formula:

$$
\boxed{
\text{knowledge flow must be conservative and heterogeneity-dissipating}
}
$$

---

# 8. The Final State Is Not Necessarily Full Consensus

A naive consensus objective would force:

$$
\widetilde\Xi_1
=
\cdots
=
\widetilde\Xi_M.
$$

This is too strong for heterogeneous graph federated learning.

Some client-specific dynamics are useful, necessary, or non-transferable.

Therefore, the desired equilibrium should instead be:

$$
\boxed{
\{\Xi_m^\star\}
=
\arg\min
\mathcal E_{\mathrm{het}}
}
$$

subject to constraints such as:

$$
\text{knowledge conservation},
$$

$$
\text{local dynamical admissibility},
$$

$$
\text{task preservation},
$$

$$
\text{stability},
$$

and possibly:

$$
\text{privacy / communication constraints}.
$$

Thus, the final state is:

> **the minimum-heterogeneity equilibrium achievable without destroying legitimate local dynamics.**

This is fundamentally different from ordinary consensus learning.

Individuality is not introduced by manually splitting a model into “shared” and “personalized” branches.

Instead:

$$
\boxed{
\text{individuality remains because the admissible dynamical constraints prevent unnecessary homogenization}
}
$$

---

# 9. Local Graph Dynamics and Federated Knowledge Flow Are Two Time Scales

The project should conceptually distinguish two dynamical processes.

### Local representation time

$$
t
$$

describes node representation evolution inside client $m$:

$$
\frac{dH_m}{dt}
=
F_m(H_m).
$$

### Federated knowledge time

$$
\tau
$$

describes cross-client redistribution of dynamical knowledge:

$$
\frac{d\widetilde\Xi_m}{d\tau}
=
J_m.
$$

The full system therefore has a two-level dynamical interpretation:

$$
\boxed{
\text{local graph flow in }t
+
\text{federated knowledge flow in }\tau.
}
$$

This distinction is important.

The federated process should not be viewed as simply modifying parameters after local optimization.

It should be understood as:

> a slower dynamical process that changes how local graph systems evolve across communication rounds.

---

# 10. The Role of the Server

At the high level, the server should not be described merely as a parameter averaging center.

Its conceptual role is:

$$
\boxed{
\text{a coordinator of conservative dynamical knowledge flow}
}
$$

The server may be responsible for estimating:

- canonical dynamical coordinates;
- global heterogeneity potential;
- aggregate conserved quantities;
- knowledge fluxes;
- admissible global constraints.

But the exact server algorithm remains open.

Possible implementations include:

- barycentric aggregation;
- pairwise flux accumulation;
- consensus Laplacians;
- operator-valued averaging;
- constrained optimization;
- secure aggregated sufficient statistics.

No particular implementation is fixed at this stage.

---

# 11. The Role of the Client

Each client should conceptually perform three tasks:

### 1. Observe local dynamics

Estimate a dynamical state:

$$
\Xi_m.
$$

### 2. Participate in conservative exchange

Transport its knowledge into a comparable space and obtain a net knowledge flux:

$$
J_m.
$$

### 3. Realize the flux locally

Map the federated knowledge flow back into the local graph dynamical system:

$$
U_m
=
\mathcal T_m^{-1}(J_m),
$$

and modify the effective local dynamics:

$$
\boxed{
\frac{dH_m}{dt}
=
F_m(H_m)
+
U_m(H_m).
}
$$

The final implementation of $U_m$ remains open.

It may eventually be realized as:

- generator correction;
- finite-time flow composition;
- initial-state transport;
- vector-field control;
- constrained residual flow;
- energy-based forcing.

Again, these are candidate realizations, not fixed components.

---

# 12. Theoretical Backbone to Pursue

The ideal theoretical chain is:

### Step 1: Define dynamical knowledge

$$
\Xi_m.
$$

### Step 2: Define cross-client transport

$$
\widetilde\Xi_m
=
\mathcal T_m(\Xi_m).
$$

### Step 3: Define heterogeneity energy

$$
\mathcal E_{\mathrm{het}}.
$$

### Step 4: Derive conservative knowledge flux

$$
J_m.
$$

such that:

$$
\sum_m p_mJ_m=0.
$$

### Step 5: Prove dissipation

$$
\frac{d\mathcal E_{\mathrm{het}}}{d\tau}
\le0.
$$

### Step 6: Characterize equilibrium

Show convergence toward:

$$
\{\Xi_m^\star\},
$$

the minimum-heterogeneity admissible dynamical state.

### Step 7: Connect federated flow back to task learning

Show or empirically verify that reducing the selected dynamical heterogeneity improves:

- representation propagation;
- transferability;
- client robustness;
- downstream accuracy.

This is the desired theory-to-algorithm chain.

---

# 13. What Is Fixed and What Is Still Open

## Fixed high-level principles

The following ideas should be treated as the current research backbone:

1. Clients are heterogeneous graph dynamical systems.
2. Federated knowledge should be dynamical rather than purely parametric.
3. Cross-client knowledge must be transported before being exchanged.
4. Knowledge exchange should obey a conservation law.
5. The flow should dissipate harmful dynamical heterogeneity.
6. The equilibrium should minimize heterogeneity under local admissibility constraints.
7. The final method should preserve useful individuality rather than force full consensus.

## Still open

The following technical choices are explicitly **not fixed**:

- exact form of dynamical knowledge $\Xi_m$;
- whether to use generator, flow, trajectory, Koopman object, or another representation;
- Functional Map or another transport mechanism;
- trajectory-induced basis construction;
- low-rank dimension;
- descriptor construction;
- server barycenter / pairwise interaction / client graph;
- exact heterogeneity energy;
- exact flux law;
- how to preserve local individuality;
- how to enforce stability;
- how the flow is injected back into A-DGN;
- whether multiple equilibria / regimes are necessary;
- whether the final method should retain FedAvg;
- exact privacy mechanism.

Any existing implementation should therefore be treated as an experimental prototype rather than the final algorithm.

---

# 14. Current Research Question

The next stage should not begin with code modification.

It should answer three mathematical questions first:

$$
\boxed{
\textbf{Q1: What exactly is conserved dynamical knowledge?}
}
$$

$$
\boxed{
\textbf{Q2: What heterogeneity energy correctly measures harmful disagreement?}
}
$$

$$
\boxed{
\textbf{Q3: What knowledge flux simultaneously satisfies conservation, dissipation, and local admissibility?}
}
$$

Once these three objects are defined coherently, the specific engineering realization can be selected afterward.

---

# 15. One-Sentence Paper Story

> **We model heterogeneous graph federated learning as a system of interacting graph dynamics, where client-specific dynamical knowledge is transported across heterogeneous function spaces and redistributed through a conservative knowledge flow that monotonically dissipates harmful dynamical heterogeneity, eventually reaching a minimum-heterogeneity equilibrium while preserving locally admissible dynamics.**

The core storyline is therefore:

$$
\boxed{
\text{Dynamics}
\rightarrow
\text{Transport}
\rightarrow
\text{Conservative Flow}
\rightarrow
\text{Heterogeneity Dissipation}
\rightarrow
\text{Admissible Equilibrium}.
}
$$
