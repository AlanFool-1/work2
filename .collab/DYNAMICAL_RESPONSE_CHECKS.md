# Dynamical response: reproducible CPU research checks

Updated: 2026-09-05 UTC

Scope: read-only numerical audit, not a trained-model experiment. Seed 20260905, CPU float64, one Torch thread; Torch 1.13.1+cu117 and PyG 2.6.1. No production source, optimizer or checkpoint changed.

## Findings

| Check | Measured result | Interpretation |
| --- | --- | --- |
| Actual ridge implementation versus identity-centered discrete ridge | `||P-I-hA||_F = 2.162e-16` | Matched objectives are a reparameterization, not different model classes |
| Native 16-step A-DGN versus composing its own step | maximum error `0` | The diagnostic uses the native computation |
| Causal response kernel, 54 observations by 4 ports | AD/central-difference relative error `4.025e-11`; future-to-past response `0` | Exact discrete causal response is measurable on this small graph |
| Finite perturbation remainder | `1.009e-3, 2.518e-4, 6.288e-5, 1.571e-5` for radii `.1, .05, .025, .0125` | Approximately quadratic remainder on this direction and untrained model |
| Toy jointly realizable conservative flow | linear conservation residual `6.360e-17`; energy rate `-0.2742308001` | The local linear equality and dissipation identity hold |
| Actual nonlinear realization | conservation drift `.0007166, .0001764, .00004374, .00001089` for steps `.1, .05, .025, .0125` | Finite updates conserve only up to nonlinear error, approximately quadratic here |
| Orthogonal client tangent ranges | zero feasible update despite disagreement norm `2.8284` | Conservativity plus local realizability need not imply consensus |

The graph is a random-initialized native model on a 12-node ring, not a trained benchmark. Observations are synthetic class-conditioned centered-logit means and second moments at steps 4, 8, 16; labels here only construct a toy sensor. Ports are initial hidden-state gain, neighbor-message impulses at steps 0 and 4, and a decay impulse at step 8. These checks do NOT validate task relevance, private-client comparability, transfer benefit, or computational feasibility of training with mixed derivatives. The analytic conservative-flow check uses a separate `tanh(B theta + b)` toy, not the graph response model.

## Reproduction

The Python block below is the complete diagnostic. Run from the repository with:

```bash
awk '/^```python$/{active=1;next} /^```$/{if(active)exit} active' .collab/DYNAMICAL_RESPONSE_CHECKS.md | /root/anaconda3/envs/torch/bin/python
```

The optional physical edge-weight probe in the first diagnostic is separate from the explicit neighbor-message impulse test; edge normalization can change the effective intervention. The impulse test fixes the intervention directly on the native message output.

## Diagnostic source

```python
"""Read-only CPU research checks; no training or checkpoint writes."""
import json
import sys
from types import SimpleNamespace

import torch
from torch_geometric.data import Data

sys.path.insert(0, '/opt/data/private/xzc/work2/backbone_functional_dynamics_stable')
from dynamics.ridge_generator import fit_ridge_generator
from models.s0.model import build_s0_model

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
torch.manual_seed(20260905)
report = {}

# The two ridge objectives in methodv0 section 10.8 are a reparameterization.
trajectory = torch.randn(17, 12, 8)
basis = torch.linalg.qr(torch.randn(12, 4), mode='reduced')[0]
dt = 0.1
lam = 1e-3
A, metrics = fit_ridge_generator(trajectory, basis, dt, lam)
z = torch.einsum('nr,tnd->trd', basis, trajectory)
X = z[:-1].permute(1, 0, 2).reshape(4, -1)
Z = z[1:].permute(1, 0, 2).reshape(4, -1)
XX = X @ X.T
reg = lam * torch.trace(XX) / 4
I = torch.eye(4)
P = torch.linalg.solve(XX + reg * I, (Z @ X.T + reg * I).T).T
equiv = float(torch.linalg.norm(P - (I + dt * A)))
assert equiv < 1e-12
report['ridge_reparameterization'] = {'frobenius_error': equiv}

# Exact finite-horizon response on the upstream native A-DGN computation.
args = SimpleNamespace(base_path='/opt/data/private/xzc/work2', n_feat=5,
    n_clss=3, hidden_dim=8, ode_steps=16, adgn_step_size=0.1,
    ode_gamma=0.1, ode_activation='tanh', enable_functional_dynamics=False)
model = build_s0_model(args).double().eval()
nodes = 12
source = torch.arange(nodes)
target = torch.roll(source, -1)
edges = torch.stack([torch.cat([source, target]), torch.cat([target, source])])
features = torch.randn(nodes, 5)
labels = torch.arange(nodes) % 3

def response(u):
    state = model.emb(features * (1 + u[0]))
    weights = torch.exp(u[1]).expand(edges.shape[1])
    saved_iters = model.conv.num_iters
    model.conv.num_iters = 1
    blocks = []
    try:
        for step in range(1, 17):
            state = model.conv(state, edges, weights)
            if step in (4, 8, 16):
                logits = model.readout(state)
                centered = logits - logits.mean(-1, keepdim=True)
                for cls in range(3):
                    values = centered[labels == cls]
                    blocks.extend([values.mean(0), values.square().mean(0)])
    finally:
        model.conv.num_iters = saved_iters
    return torch.cat(blocks)

zero = torch.zeros(2)
base = response(zero)
K = torch.autograd.functional.jacobian(response, zero)
direction = torch.tensor([0.6, 0.8])
derivative = K @ direction
step = 1e-5
finite_difference = (response(step * direction) - response(-step * direction)) / (2 * step)
relative_error = float(torch.linalg.norm(finite_difference - derivative) / torch.linalg.norm(derivative))
assert relative_error < 1e-7
remainders = []
for radius in [0.1, 0.05, 0.025, 0.0125]:
    error = float(torch.linalg.norm(response(radius * direction) - base - radius * derivative))
    remainders.append({'radius': radius, 'remainder': error, 'remainder_over_radius_squared': error / radius**2})
report['native_adgn_response'] = {'shape': list(K.shape), 'jvp_central_difference_relative_error': relative_error,
    'taylor_remainders': remainders}

# Native forward remains identical to composing the same Euler step 16 times.
data = Data(x=features, edge_index=edges, edge_weight=torch.ones(edges.shape[1]))
native = model(data)
state = model.emb(features)
model.conv.num_iters = 1
for _ in range(16):
    state = model.conv(state, edges, data.edge_weight)
model.conv.num_iters = 16
native_error = float((model.readout(state) - native).abs().max())
assert native_error == 0
report['native_forward_max_error'] = native_error

# Time-resolved physical ports: initial gain, neighbor impulses at s=0,4,
# and an additive decay impulse at s=8. These are dynamics interventions.
def impulse_response(u):
    state = model.emb(features) * (1 + u[0])
    saved_iters = model.conv.num_iters
    model.conv.num_iters = 1
    blocks = []
    try:
        for s in range(16):
            gain = u[1] if s == 0 else u[2] if s == 4 else u.new_zeros(())
            handle = model.conv.gcn_conv.register_forward_hook(
                lambda module, inputs, output, gain=gain: output * (1 + gain))
            try:
                next_state = model.conv(state, edges, torch.ones(edges.shape[1]))
            finally:
                handle.remove()
            if s == 8:
                next_state = next_state - model.conv.epsilon * u[3] * state
            state = next_state
            if s + 1 in (4, 8, 16):
                logits = model.readout(state)
                centered = logits - logits.mean(-1, keepdim=True)
                for cls in range(3):
                    values = centered[labels == cls]
                    blocks.extend([values.mean(0), values.square().mean(0)])
    finally:
        model.conv.num_iters = saved_iters
    return torch.cat(blocks)

zero_impulse = torch.zeros(4)
impulse_base = impulse_response(zero_impulse)
kernel = torch.autograd.functional.jacobian(impulse_response, zero_impulse)
assert torch.equal(impulse_base, base)
causal_error = max(float(kernel[:18, 2:].abs().max()), float(kernel[:36, 3].abs().max()))
assert causal_error == 0
impulse_direction = torch.tensor([1.0, -1.0, 1.0, 1.0]) / 2
impulse_derivative = kernel @ impulse_direction
central = (impulse_response(step * impulse_direction) - impulse_response(-step * impulse_direction)) / (2 * step)
impulse_jvp_error = float(torch.linalg.norm(central - impulse_derivative) / torch.linalg.norm(impulse_derivative))
assert impulse_jvp_error < 1e-7
impulse_errors = []
for radius in [0.1, 0.05, 0.025, 0.0125]:
    error = float(torch.linalg.norm(impulse_response(radius * impulse_direction) - impulse_base - radius * impulse_derivative))
    impulse_errors.append({'radius': radius, 'remainder': error, 'remainder_over_radius_squared': error / radius**2})
report['causal_dynamical_response_kernel'] = {'shape': list(kernel.shape),
    'future_to_past_max_response': causal_error, 'central_difference_relative_error': impulse_jvp_error,
    'taylor_remainders': impulse_errors}

# Analytic feasible conservative response flow, with nonlinear realization.
p = torch.tensor([0.2, 0.3, 0.5])
B = torch.randn(3, 4, 5) / 2
theta = torch.randn(3, 5) / 3
b = torch.randn(3, 4) / 3
def observed(par):
    return torch.tanh(torch.einsum('mdk,mk->md', B, par) + b)
x = observed(theta)
mean = torch.einsum('m,md->d', p, x)
e = x - mean
D = (1 - x.square()).unsqueeze(-1) * B
mobility = D @ D.transpose(-1, -2)
dual = torch.linalg.solve(torch.einsum('m,mde->de', p, mobility),
    torch.einsum('m,mde,me->d', p, mobility, e))
v = -torch.einsum('mdk,md->mk', D, e - dual)
flux = torch.einsum('mdk,mk->md', D, v)
conservation = float(torch.linalg.norm(torch.einsum('m,md->d', p, flux)))
rate = float(torch.einsum('m,md,md->', p, e, flux))
dissipation = -float(torch.einsum('m,mk,mk->', p, v, v))
assert conservation < 1e-12
assert abs(rate - dissipation) < 1e-12
def energy(xx):
    avg = torch.einsum('m,md->d', p, xx)
    return float(0.5 * torch.einsum('m,md,md->', p, xx - avg, xx - avg))
updates = []
for alpha in [0.1, 0.05, 0.025, 0.0125]:
    plus = observed(theta + alpha * v)
    drift = float(torch.linalg.norm(torch.einsum('m,md->d', p, plus - x)))
    updates.append({'step': alpha, 'actual_conservation_drift': drift,
        'drift_over_step_squared': drift / alpha**2, 'energy_change': energy(plus) - energy(x)})
report['realizable_conservative_flow'] = {'linear_conservation_residual': conservation,
    'energy_rate': rate, 'negative_update_norm_squared': dissipation, 'nonlinear_steps': updates}

# Incompatible local tangent ranges can forbid any conservative movement.
D1 = torch.tensor([[1.0], [0.0]])
D2 = torch.tensor([[0.0], [1.0]])
e1 = torch.tensor([1.0, -1.0])
e2 = -e1
H1, H2 = D1 @ D1.T, D2 @ D2.T
mu = torch.linalg.solve(H1 + H2, H1 @ e1 + H2 @ e2)
v1, v2 = -D1.T @ (e1 - mu), -D2.T @ (e2 - mu)
assert float(v1.abs().max() + v2.abs().max()) == 0
report['incompatible_tangent_ranges'] = {'update_norm': 0.0, 'response_disagreement_norm': float(torch.linalg.norm(e1-e2))}
print(json.dumps(report, indent=2))

```
