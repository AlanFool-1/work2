from typing import Dict, Optional

import torch
from torch.nn import functional as F

from .observable import KoopmanObservable


def manifold_closure_loss(
    trajectory: torch.Tensor,
    observable: KoopmanObservable,
    exclude_initial: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Relative observable-manifold closure error.

    The old raw MSE could look numerically tiny simply because the auxiliary
    coordinates had small scale.  v1.1 normalizes by target auxiliary energy.
    """
    states = trajectory[1:] if exclude_initial else trajectory
    if states.shape[0] == 0:
        return trajectory.new_zeros(())
    h = states[..., : observable.hidden_dim]
    aux = states[..., observable.hidden_dim :]
    target = observable.aux(h)

    num = (aux - target).square().sum(dim=-1)
    den = target.square().sum(dim=-1).clamp_min(eps)
    return (num / den).mean()


def node_classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    train_mask: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    if train_mask is not None:
        logits = logits[train_mask]
        labels = labels[train_mask]
    return F.cross_entropy(logits, labels, label_smoothing=float(label_smoothing))


def method1_local_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    trajectory: torch.Tensor,
    observable: KoopmanObservable,
    manifold_weight: float,
    train_mask: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
    koopman_field=None,
    operator_speed_weight: float = 0.0,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    task = node_classification_loss(
        logits,
        labels,
        train_mask=train_mask,
        label_smoothing=label_smoothing,
    )
    closure = manifold_closure_loss(trajectory, observable)

    if koopman_field is not None and operator_speed_weight > 0:
        speed = koopman_field.operator_speed_penalty()
    else:
        speed = task.new_zeros(())

    total = (
        task
        + float(manifold_weight) * closure
        + float(operator_speed_weight) * speed
    )
    parts = {
        "loss": total.detach(),
        "task_loss": task.detach(),
        "manifold_loss": closure.detach(),
        "operator_speed_loss": speed.detach(),
    }
    return total, parts
