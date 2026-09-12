"""Shared O4 block-v FedAdamW optimizer-state protocol."""

from __future__ import annotations

import math

import numpy as np
import torch


class FederatedAdamW(torch.optim.Optimizer):
    """AdamW with round-local first moment and cross-round block-v state."""

    def __init__(
        self,
        named_params,
        lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        global_v_step=0,
    ):
        named_params = list(named_params)
        if not named_params:
            raise ValueError('FederatedAdamW requires parameters.')
        names = [name for name, _ in named_params]
        if len(names) != len(set(names)):
            raise ValueError('FederatedAdamW parameter names must be unique.')
        if lr <= 0.0 or eps <= 0.0 or weight_decay < 0.0:
            raise ValueError('Invalid FederatedAdamW hyperparameters.')
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError('FederatedAdamW betas must lie in [0, 1).')
        if int(global_v_step) < 0:
            raise ValueError('global_v_step must be non-negative.')

        defaults = {
            'lr': float(lr),
            'betas': tuple(float(value) for value in betas),
            'eps': float(eps),
            'weight_decay': float(weight_decay),
        }
        super().__init__([parameter for _, parameter in named_params], defaults)
        self.global_v_step_start = int(global_v_step)
        self.last_adaptive_step_rms = 0.0
        self.last_bias_correction1 = 0.0
        self.last_bias_correction2 = 0.0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        adaptive_sq = 0.0
        adaptive_count = 0
        for group in self.param_groups:
            lr = float(group['lr'])
            beta1, beta2 = group['betas']
            eps = float(group['eps'])
            weight_decay = float(group['weight_decay'])
            for parameter in group['params']:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError(
                        'FederatedAdamW does not support sparse gradients.'
                    )
                if not torch.isfinite(gradient).all():
                    raise FloatingPointError(
                        'FederatedAdamW received a non-finite gradient.'
                    )

                state = self.state[parameter]
                if not state:
                    state['exp_avg'] = torch.zeros_like(parameter)
                    state['exp_avg_sq'] = torch.zeros_like(parameter)
                    state['local_m_step'] = 0
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                state['local_m_step'] += 1
                local_step = int(state['local_m_step'])
                global_step = self.global_v_step_start + local_step

                exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(
                    gradient, gradient, value=1.0 - beta2
                )
                bias_correction1 = 1.0 - beta1 ** local_step
                bias_correction2 = 1.0 - beta2 ** global_step
                adaptive_direction = (
                    exp_avg / bias_correction1
                ) / ((exp_avg_sq / bias_correction2).sqrt().add(eps))

                if weight_decay:
                    parameter.mul_(1.0 - lr * weight_decay)
                parameter.add_(adaptive_direction, alpha=-lr)

                adaptive_sq += float(
                    adaptive_direction.square().sum().detach().cpu().item()
                ) * (lr ** 2)
                adaptive_count += int(parameter.numel())
                self.last_bias_correction1 = float(bias_correction1)
                self.last_bias_correction2 = float(bias_correction2)

        self.last_adaptive_step_rms = math.sqrt(
            adaptive_sq / max(adaptive_count, 1)
        )
        return loss


@torch.no_grad()
def initialize_second_moment_blocks(optimizer, named_params, block_stats):
    """Expand one non-negative server scalar into each parameter tensor."""

    for name, parameter in named_params:
        scalar = float(block_stats.get(name, 0.0))
        if not math.isfinite(scalar) or scalar < -1e-12:
            raise ValueError(f'Invalid block second moment for {name}: {scalar}')
        state = optimizer.state[parameter]
        state['exp_avg'] = torch.zeros_like(parameter)
        state['local_m_step'] = 0
        state['exp_avg_sq'] = torch.full_like(
            parameter, fill_value=max(scalar, 0.0)
        )


@torch.no_grad()
def export_block_second_moment(optimizer, named_params):
    """Compress each main-parameter second moment to one mean scalar."""

    exported = {}
    for name, parameter in named_params:
        state = optimizer.state.get(parameter, {})
        value = state.get('exp_avg_sq')
        exported[name] = (
            0.0 if value is None else float(value.mean().detach().cpu().item())
        )
    return exported


@torch.no_grad()
def optimizer_state_diagnostics(optimizer):
    m_sq = 0.0
    v_sq = 0.0
    v_sum = 0.0
    count = 0
    for state in optimizer.state.values():
        if 'exp_avg' not in state or 'exp_avg_sq' not in state:
            continue
        exp_avg = state['exp_avg']
        exp_avg_sq = state['exp_avg_sq']
        m_sq += float(exp_avg.square().sum().detach().cpu().item())
        v_sq += float(exp_avg_sq.square().sum().detach().cpu().item())
        v_sum += float(exp_avg_sq.sum().detach().cpu().item())
        count += int(exp_avg.numel())
    denominator = max(count, 1)
    return {
        'main_m_rms': math.sqrt(m_sq / denominator),
        'main_v_rms': math.sqrt(v_sq / denominator),
        'main_v_block_mean': v_sum / denominator,
        'main_adaptive_step_rms': float(
            getattr(optimizer, 'last_adaptive_step_rms', 0.0)
        ),
    }


def _validate_weights(payloads, weights):
    if not payloads or len(payloads) != len(weights):
        raise ValueError('Payload/weight count mismatch.')
    numeric = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(numeric).all() or np.any(numeric < 0.0):
        raise ValueError('Aggregation weights must be finite and non-negative.')
    if not np.isclose(float(numeric.sum()), 1.0):
        raise ValueError('Aggregation weights must sum to one.')
    return numeric.tolist()


def aggregate_block_second_moments(payloads, weights):
    """Apply the model aggregation weights to schema-identical block-v payloads."""

    weights = _validate_weights(payloads, weights)
    names = set(payloads[0])
    if any(set(payload) != names for payload in payloads[1:]):
        raise ValueError('Block-v payloads have inconsistent parameter names.')
    aggregated = {}
    for name in sorted(names):
        value = 0.0
        for weight, payload in zip(weights, payloads):
            scalar = float(payload[name])
            if not math.isfinite(scalar) or scalar < -1e-12:
                raise ValueError(f'Invalid block-v payload for {name}.')
            value += float(weight) * max(scalar, 0.0)
        aggregated[name] = value
    return aggregated


def mapping_mean(mapping):
    if not mapping:
        return 0.0
    values = np.asarray(list(mapping.values()), dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < -1e-12):
        raise ValueError('Block-v mapping must be finite and non-negative.')
    return float(values.mean())
