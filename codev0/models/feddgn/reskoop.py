"""Passive ResKoopNet identification for detached FedDGN trajectories.

The task backbone never receives gradients from this module. ResKoopNet
learns only its observable dictionary, while the discrete Koopman operator is
recomputed by ridge-regularized EDMD during every outer optimization step.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn


KOOPMAN_DICTIONARY_PROTOCOL = 'reskoop_dictionary_fedavg_v1'


class ResKoopDictionary(nn.Module):
    """Observable dictionary ``Psi(h) = [1, h, psi_theta(h)]``."""

    def __init__(
        self,
        state_dim: int,
        aux_dim: int = 16,
        hidden_dim: int = 64,
        hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.aux_dim = int(aux_dim)
        self.observable_dim = 1 + self.state_dim + self.aux_dim

        if self.aux_dim == 0:
            self.network = None
            return

        layers = []
        input_dim = self.state_dim
        for _ in range(int(hidden_layers)):
            layers.extend([nn.Linear(input_dim, hidden_dim), nn.Tanh()])
            input_dim = int(hidden_dim)
        layers.append(nn.Linear(input_dim, self.aux_dim))
        self.network = nn.Sequential(*layers)

    def learned_features(self, state: torch.Tensor) -> torch.Tensor:
        if self.network is None:
            return state.new_empty(state.size(0), 0)
        return self.network(state)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        constant = torch.ones_like(state[:, :1])
        return torch.cat(
            [constant, state, self.learned_features(state)], dim=-1
        )


class PassiveResKoopNet(nn.Module):
    """Learn a dictionary and identify Koopman modes from a detached rollout."""

    def __init__(
        self,
        state_dim: int,
        aux_dim: int = 16,
        hidden_dim: int = 64,
        hidden_layers: int = 2,
        ridge: float = 1e-3,
        max_snapshots: int = 8192,
    ) -> None:
        super().__init__()
        self.dictionary = ResKoopDictionary(
            state_dim=state_dim,
            aux_dim=aux_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
        )
        self.ridge = float(ridge)
        self.max_snapshots = int(max_snapshots)

    @property
    def state_dim(self) -> int:
        return self.dictionary.state_dim

    @property
    def observable_dim(self) -> int:
        return self.dictionary.observable_dim

    def snapshot_pairs(self, trajectory) -> Tuple[torch.Tensor, torch.Tensor]:
        """Flatten adjacent trajectory states into node-time snapshot pairs."""

        if isinstance(trajectory, (tuple, list)):
            trajectory = torch.stack(tuple(trajectory), dim=0)

        current = trajectory[:-1].reshape(-1, self.state_dim)
        following = trajectory[1:].reshape(-1, self.state_dim)
        if current.size(0) > self.max_snapshots:
            # Deterministic even spacing covers all rollout times and avoids
            # consuming the task optimizer's RNG stream.
            indices = torch.linspace(
                0,
                current.size(0) - 1,
                steps=self.max_snapshots,
                device=current.device,
            ).round().long()
            current = current.index_select(0, indices)
            following = following.index_select(0, indices)
        return current.detach(), following.detach()

    def _operator(self, psi_x: torch.Tensor, psi_y: torch.Tensor):
        gram = psi_x.transpose(0, 1) @ psi_x
        cross = psi_x.transpose(0, 1) @ psi_y
        identity = torch.eye(
            gram.size(0), device=gram.device, dtype=gram.dtype
        )
        regularized = gram + self.ridge * identity
        try:
            operator = torch.linalg.solve(regularized, cross)
        except RuntimeError:
            operator = torch.linalg.pinv(regularized) @ cross

        symmetric_gram = 0.5 * (gram + gram.transpose(0, 1))
        gram_eigenvalues = torch.linalg.eigvalsh(symmetric_gram)
        largest = gram_eigenvalues[-1].clamp_min(0.0)
        smallest = gram_eigenvalues[0].clamp_min(0.0)
        tolerance = (
            torch.finfo(gram.dtype).eps
            * max(psi_x.shape)
            * largest.clamp_min(torch.finfo(gram.dtype).tiny)
        )
        effective_rank = (gram_eigenvalues > tolerance).sum()
        condition = (largest + self.ridge) / (
            smallest + self.ridge
        ).clamp_min(torch.finfo(gram.dtype).tiny)
        return operator, effective_rank, condition

    @staticmethod
    def _spectral_terms(operator: torch.Tensor):
        eigenvalues, eigenvectors = torch.linalg.eig(operator)
        order = torch.argsort(eigenvalues.abs(), descending=True)
        return (
            eigenvalues.index_select(0, order).detach(),
            eigenvectors.index_select(1, order).detach(),
        )

    @staticmethod
    def _mode_residuals(
        psi_x: torch.Tensor,
        psi_y: torch.Tensor,
        eigenvalues: torch.Tensor,
        eigenvectors: torch.Tensor,
    ) -> torch.Tensor:
        complex_dtype = eigenvectors.dtype
        phi_x = psi_x.to(complex_dtype) @ eigenvectors
        phi_y = psi_y.to(complex_dtype) @ eigenvectors
        residual = phi_y - phi_x * eigenvalues.unsqueeze(0)
        numerator = torch.linalg.vector_norm(residual, dim=0)
        denominator = torch.linalg.vector_norm(phi_y, dim=0).clamp_min(1e-12)
        return numerator / denominator

    @classmethod
    def _spectral_loss(
        cls,
        psi_x: torch.Tensor,
        psi_y: torch.Tensor,
        eigenvalues: torch.Tensor,
        eigenvectors: torch.Tensor,
    ) -> torch.Tensor:
        # A mean over normalized per-mode residuals prevents high-energy modes
        # from hiding weak but inaccurate modes.
        return cls._mode_residuals(
            psi_x, psi_y, eigenvalues, eigenvectors
        ).square().mean()

    @staticmethod
    def _relative_norm(
        numerator: torch.Tensor, denominator: torch.Tensor
    ) -> torch.Tensor:
        return torch.linalg.vector_norm(numerator) / torch.linalg.vector_norm(
            denominator
        ).clamp_min(1e-12)

    def _koopman_modes(self, eigenvectors: torch.Tensor) -> torch.Tensor:
        """Return state-space modes with shape [state_dim, observable_dim]."""

        selector = torch.zeros(
            self.observable_dim,
            self.state_dim,
            device=eigenvectors.device,
            dtype=eigenvectors.dtype,
        )
        selector[1:1 + self.state_dim] = torch.eye(
            self.state_dim,
            device=eigenvectors.device,
            dtype=eigenvectors.dtype,
        )
        inverse = torch.linalg.pinv(eigenvectors)
        return (inverse @ selector).transpose(0, 1)

    def _identify(self, current: torch.Tensor, following: torch.Tensor):
        psi_x = self.dictionary(current)
        psi_y = self.dictionary(following)
        operator, effective_rank, condition = self._operator(psi_x, psi_y)
        eigenvalues, eigenvectors = self._spectral_terms(operator)
        fit_residual = self._relative_norm(psi_y - psi_x @ operator, psi_y)
        mode_residuals = self._mode_residuals(
            psi_x, psi_y, eigenvalues, eigenvectors
        )
        spectral_residual = torch.sqrt(mode_residuals.square().mean())
        modes = self._koopman_modes(eigenvectors)
        learned = self.dictionary.learned_features(current)
        aux_rms = (
            torch.sqrt(learned.square().mean())
            if learned.numel()
            else current.new_zeros(())
        )
        diagnostics = {
            'koopman_fit_residual': fit_residual,
            'koopman_spectral_residual': spectral_residual,
            'koopman_spectral_radius': eigenvalues.abs().max(),
            'koopman_operator_fro': torch.linalg.matrix_norm(operator),
            'koopman_condition': condition,
            'koopman_effective_rank': effective_rank.to(current.dtype),
            'koopman_aux_rms': aux_rms,
            'koopman_snapshots': current.new_tensor(float(current.size(0))),
            'koopman_mode_residual_mean': mode_residuals.mean(),
            'koopman_mode_residual_max': mode_residuals.max(),
        }
        return {
            'operator': operator,
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'mode_residuals': mode_residuals,
            'modes': modes,
            'diagnostics': diagnostics,
        }

    def fit_trajectory(
        self,
        trajectory,
        optimizer: Optional[torch.optim.Optimizer],
        outer_steps: int = 1,
        inner_steps: int = 4,
        batch_size: int = 1024,
        grad_clip: float = 5.0,
        sample_seed: int = 0,
    ) -> Dict[str, object]:
        """Fit the dictionary and return a detached Koopman analysis bundle."""

        current, following = self.snapshot_pairs(trajectory)
        self.train()

        dictionary_loss = current.new_zeros(())
        generator = torch.Generator(device='cpu')
        generator.manual_seed(int(sample_seed))
        if optimizer is not None and outer_steps > 0 and inner_steps > 0:
            for _ in range(int(outer_steps)):
                with torch.no_grad():
                    identified = self._identify(current, following)
                    eigenvalues = identified['eigenvalues']
                    eigenvectors = identified['eigenvectors']

                for _ in range(int(inner_steps)):
                    count = min(int(batch_size), current.size(0))
                    indices = torch.randperm(
                        current.size(0), generator=generator, device='cpu'
                    )[:count].to(current.device)
                    batch_x = current.index_select(0, indices)
                    batch_y = following.index_select(0, indices)
                    optimizer.zero_grad(set_to_none=True)
                    dictionary_loss = self._spectral_loss(
                        self.dictionary(batch_x),
                        self.dictionary(batch_y),
                        eigenvalues,
                        eigenvectors,
                    )
                    if not torch.isfinite(dictionary_loss):
                        raise FloatingPointError(
                            'Non-finite ResKoop dictionary loss.'
                        )
                    dictionary_loss.backward()
                    if grad_clip > 0.0:
                        torch.nn.utils.clip_grad_norm_(
                            self.dictionary.parameters(),
                            grad_clip,
                            error_if_nonfinite=True,
                        )
                    optimizer.step()

        self.eval()
        with torch.no_grad():
            result = self._identify(current, following)
        result['diagnostics']['koopman_dictionary_loss'] = (
            dictionary_loss.detach()
        )
        for name, value in result['diagnostics'].items():
            if not torch.isfinite(value).all():
                raise FloatingPointError(
                    f'Non-finite ResKoop diagnostic {name}.'
                )
        return {
            name: (
                value.detach()
                if torch.is_tensor(value)
                else {
                    key: item.detach() for key, item in value.items()
                }
            )
            for name, value in result.items()
        }


def build_passive_reskoop(args) -> PassiveResKoopNet:
    return PassiveResKoopNet(
        state_dim=int(args.feddgn_hidden_dim),
        aux_dim=int(args.feddgn_koopman_aux_dim),
        hidden_dim=int(args.feddgn_koopman_hidden_dim),
        hidden_layers=int(args.feddgn_koopman_hidden_layers),
        ridge=float(args.feddgn_koopman_ridge),
        max_snapshots=int(args.feddgn_koopman_max_snapshots),
    )


__all__ = [
    'KOOPMAN_DICTIONARY_PROTOCOL',
    'PassiveResKoopNet',
    'ResKoopDictionary',
    'build_passive_reskoop',
]
