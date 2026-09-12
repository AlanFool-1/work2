from typing import Callable, List

import torch


VectorField = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def euler_step(func: VectorField, t: torch.Tensor, z: torch.Tensor, dt: float) -> torch.Tensor:
    return z + dt * func(t, z)


def heun_step(func: VectorField, t: torch.Tensor, z: torch.Tensor, dt: float) -> torch.Tensor:
    k1 = func(t, z)
    pred = z + dt * k1
    k2 = func(t + dt, pred)
    return z + 0.5 * dt * (k1 + k2)


def rk4_step(func: VectorField, t: torch.Tensor, z: torch.Tensor, dt: float) -> torch.Tensor:
    half = 0.5 * dt
    k1 = func(t, z)
    k2 = func(t + half, z + half * k1)
    k3 = func(t + half, z + half * k2)
    k4 = func(t + dt, z + dt * k3)
    return z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate_fixed(
    func: VectorField,
    z0: torch.Tensor,
    horizon: float,
    steps: int,
    solver: str = "rk4",
    return_trajectory: bool = True,
) -> torch.Tensor:
    """Differentiable fixed-step ODE integration implemented with PyTorch only."""

    if steps <= 0:
        raise ValueError("steps must be positive")
    dt = float(horizon) / float(steps)
    t = z0.new_tensor(0.0)
    z = z0
    trajectory: List[torch.Tensor] = [z0]

    if solver == "euler":
        step_fn = euler_step
    elif solver == "heun":
        step_fn = heun_step
    elif solver == "rk4":
        step_fn = rk4_step
    else:
        raise ValueError(f"Unknown solver={solver}")

    for _ in range(steps):
        z = step_fn(func, t, z, dt)
        t = t + dt
        if return_trajectory:
            trajectory.append(z)

    if return_trajectory:
        return torch.stack(trajectory, dim=0)
    return z
