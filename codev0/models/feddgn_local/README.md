# FedDGN v0: local intrinsic dynamics first

This package is the clean official-style A-DGN backbone used after the PH-DGN /
SONAR trials. Its defaults restore the previously validated federated baseline,
except that the current experiment deliberately keeps one local epoch per round.

## 1. Intrinsic local dynamics

The recurrent state update follows the A-DGN discretization (Gravina, Bacciu,
Gallicchio, ICLR 2023):

```text
H_{l+1} = H_l + eps * sigma(
    (W - W^T - gamma I) H_l + Phi(H_l, G) + b
)
```

`Phi` is implemented by the released normalized-GCN variant. Its default
self-loop handling is preserved to match the original A-DGN implementation.
The same parameters are shared across all recurrent steps. The default rollout
uses the previously validated `16` Euler steps; increasing the rollout depth does
not introduce a separate parameter set per step.

## 2. External-force port (reserved, inactive now)

The model exposes

```python
model(data, external_force=None, force_scale=0.0)
model.compute_vector_field(state, data, external_force=...)
model.forward_with_aux(data, external_force=...)
```

and evolves conceptually as

```text
dH/dt = F_intrinsic(H, G) + beta * F_external.
```

For the current local stage, training and validation always pass
`external_force=None` and `force_scale=0.0`.

The future force can be either:

```python
# Tensor force
force = torch.Tensor[num_nodes, hidden_dim]

# State-dependent callable force
force = lambda state, intrinsic, step_id, data: ...
```

This combines the tensor/callable port pattern used in the PH-DGN adaptation
with the state-dependent forcing semantics used in SONAR / port-Hamiltonian
simulators.  A later Koopman/TANGO module can therefore be attached without
rewriting the intrinsic local dynamics.

## 3. Local-only federated lifecycle

`client.py` deliberately keeps communication and aggregation alive but comments
out the FedAvg broadcast installation:

```python
# set_state_dict(self.model, self.sd['global'], self.gpu_id)
```

Thus:

- clients still upload model states every round;
- the server still runs its ordinary `FullModelServer` aggregation;
- each client ignores the aggregate and keeps its own local parameters;
- Adam state is persistent across rounds.

When switching back to FedAvg later, uncomment the line above and rebuild the
optimizer after installing the global model.

## 4. Koopman-ready interface

`forward_with_aux` returns

```text
h0, ht, trajectory=(h0, h1, ..., hT)
```

so a later ResKoopNet module can consume the trajectory directly.  No Koopman
loss or communication is included in this version.

## 5. Optional arguments

No new parser arguments are required; every item below has a default and is
read with `getattr`.

```text
feddgn_hidden_dim          default: 64
feddgn_steps               default: 16
feddgn_step_size           default: 0.1
feddgn_gamma               default: 0.1
feddgn_activation          default: tanh
feddgn_input_dropout       default: 0.0
feddgn_classifier_dropout  default: 0.0
feddgn_head                default: linear   (mlp | linear)
feddgn_lr                  fallback: lr_main -> base_lr -> lr -> 1.5e-2
feddgn_label_smoothing     default: 0.0
feddgn_max_grad_norm       default: 0.0 (disabled)
```

If the outer project hard-codes allowed model names, add `feddgn_local` there. The
package itself follows the existing `models/<model_name>/Client, Server`
interface.
