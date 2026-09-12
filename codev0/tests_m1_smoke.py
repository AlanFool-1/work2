"""Executable checks for the Method 1.1 Port-Hamiltonian backbone.

Run from ``codev0`` with ``python -m pytest tests_m1_smoke.py -q``. These cover
the two properties every later v11 experiment leans on: the per-band generator
is actually dissipative, and the manifold closure is a relative error.
"""

import torch

from models.v11 import Method1LocalConfig, Method1LocalKoopmanGNN, method1_local_loss


def _ring(n: int) -> torch.Tensor:
    src = torch.arange(n)
    dst = (src + 1) % n
    return torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)


def _config(**overrides) -> Method1LocalConfig:
    values = dict(
        in_dim=7,
        num_classes=3,
        hidden_dim=12,
        aux_dim=6,
        observable_hidden_dim=16,
        encoder_hidden_dim=16,
        basis_degree=2,
        operator_rank=3,
        ode_steps=3,
        input_dropout=0.0,
        encoder_dropout=0.0,
        classifier_dropout=0.0,
    )
    values.update(overrides)
    return Method1LocalConfig(**values)


def test_forward_backward_and_ph_certificate() -> None:
    torch.manual_seed(0)
    n = 24
    cfg = _config()
    model = Method1LocalKoopmanGNN(cfg)
    x = torch.randn(n, 7)
    y = torch.randint(3, (n,))
    out = model(x, _ring(n), return_details=True)
    assert out["logits"].shape == (n, 3)
    assert out["trajectory"].shape == (cfg.ode_steps + 1, n, cfg.observable_dim)

    loss, parts = method1_local_loss(
        out["logits"],
        y,
        out["trajectory"],
        model.observable,
        cfg.manifold_weight,
        label_smoothing=cfg.label_smoothing,
        koopman_field=model.koopman_field,
        operator_speed_weight=cfg.operator_speed_weight,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None for p in model.parameters())

    # G_q = alpha_q J - beta_q R - gamma_q I, so sym(G_q) = -beta_q R - gamma_q I
    # is negative semidefinite with max eigenvalue at most -gamma_q.
    for q in range(cfg.basis_degree + 1):
        G = model.koopman_field.dense_generator(q).detach()
        sym = 0.5 * (G + G.T)
        max_eig = float(torch.linalg.eigvalsh(sym).max())
        gamma = float(model.koopman_field.band_coeffs[q].gamma)
        assert max_eig <= -gamma + 1e-6, (max_eig, gamma)

    # beta and gamma are squared parameters, so they can reach zero but never
    # go below it regardless of how the raw parameters move.
    for coeff in model.koopman_field.band_coeffs:
        with torch.no_grad():
            coeff.beta_raw.fill_(-3.0)
            coeff.gamma_raw.fill_(-3.0)
        assert float(coeff.beta) > 0 and float(coeff.gamma) >= cfg.min_gamma


def test_manifold_closure_is_scale_relative() -> None:
    """Rescaling the auxiliary observable must not change the closure error."""
    torch.manual_seed(2)
    cfg = _config()
    model = Method1LocalKoopmanGNN(cfg)
    x = torch.randn(16, 7)
    out = model(x, _ring(16), return_details=True)
    scaled = _config()
    model_scaled = Method1LocalKoopmanGNN(scaled)
    model_scaled.load_state_dict(model.state_dict())
    model_scaled.observable.aux_scale = 1000.0

    trajectory = out["trajectory"].detach()
    reference = float(
        method1_local_loss(
            out["logits"], torch.zeros(16, dtype=torch.long), trajectory,
            model.observable, 0.0,
        )[1]["manifold_loss"]
    )
    scaled_trajectory = trajectory.clone()
    scaled_trajectory[..., cfg.hidden_dim :] *= 1000.0
    rescaled = float(
        method1_local_loss(
            out["logits"], torch.zeros(16, dtype=torch.long), scaled_trajectory,
            model_scaled.observable, 0.0,
        )[1]["manifold_loss"]
    )
    assert abs(reference - rescaled) < 1e-6, (reference, rescaled)


def test_operator_payload_roundtrip() -> None:
    torch.manual_seed(1)
    cfg = _config(in_dim=5, num_classes=2, hidden_dim=8, aux_dim=4, basis_degree=1,
                  operator_rank=2, ode_steps=2)
    m1 = Method1LocalKoopmanGNN(cfg)
    m2 = Method1LocalKoopmanGNN(cfg)
    m2.load_operator_state(m1.export_operator_state(cpu=True))
    for a, b in zip(m1.dense_koopman_generators(), m2.dense_koopman_generators()):
        assert torch.allclose(a, b)
