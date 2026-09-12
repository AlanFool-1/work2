from typing import Dict, List

import torch

from .model import Method1LocalKoopmanGNN


@torch.no_grad()
def generator_diagnostics(model: Method1LocalKoopmanGNN) -> List[Dict[str, float]]:
    """Per-band PH/Koopman diagnostics."""
    field = model.koopman_field
    results: List[Dict[str, float]] = []
    for q in range(field.basis_degree + 1):
        coeff = field.band_coeffs[q]
        core = field.core_for_band(q)
        G = field.dense_generator(q)
        J = core.dense_J()
        R = core.dense_R()
        sym = 0.5 * (G + G.T)
        sym_eigs = torch.linalg.eigvalsh(sym)
        singular = torch.linalg.svdvals(G)
        energy = singular.square()
        cumulative = torch.cumsum(energy, dim=0) / energy.sum().clamp_min(1e-12)
        eff90 = int((cumulative < 0.90).sum().item() + 1)
        results.append(
            {
                "band": float(q),
                "alpha": float(coeff.alpha.item()),
                "beta": float(coeff.beta.item()),
                "gamma": float(coeff.gamma.item()),
                "J_fro": float(torch.linalg.norm(J).item()),
                "R_fro": float(torch.linalg.norm(R).item()),
                "sym_max_eig": float(sym_eigs.max().item()),
                "sym_min_eig": float(sym_eigs.min().item()),
                "spectral_norm": float(singular.max().item()),
                "rank90_energy": float(eff90),
            }
        )
    return results


@torch.no_grad()
def trajectory_diagnostics(details: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Within-forward norm diagnostics.

    Port-Hamiltonian stability constrains norm change *along one trajectory*.
    It does not forbid the encoder from changing its output scale across epochs.
    """
    traj = details["trajectory"]
    norms = torch.linalg.vector_norm(traj.reshape(traj.shape[0], -1), dim=1)
    z0 = norms[0].clamp_min(1e-12)
    hT = details["hT"]
    auxT = details["auxT"]
    return {
        "z0_norm": float(norms[0].item()),
        "zT_norm": float(norms[-1].item()),
        "trajectory_growth_ratio": float((norms[-1] / z0).item()),
        "trajectory_max_ratio": float((norms.max() / z0).item()),
        "hT_rms": float(hT.square().mean().sqrt().item()),
        "auxT_rms": float(auxT.square().mean().sqrt().item()),
    }


def parameter_report(model: Method1LocalKoopmanGNN) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    koopman = sum(p.numel() for p in model.koopman_field.parameters())
    encoder = sum(p.numel() for p in model.encoder.parameters())
    observable = sum(p.numel() for p in model.observable.parameters())
    classifier = sum(p.numel() for p in model.classifier.parameters())
    return {
        "total": total,
        "koopman": koopman,
        "encoder": encoder,
        "observable": observable,
        "classifier": classifier,
    }
