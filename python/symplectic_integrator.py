"""
Symplectic integrators for Hamiltonian Neural Networks.

Symplectic integrators preserve the phase-space structure (Liouville's theorem)
and provide bounded energy error (no secular drift), making them ideal for
long-horizon predictions with HNNs.

Implemented integrators:
    - Leapfrog (Stormer-Verlet): 2nd order, symplectic
    - Euler (non-symplectic, for comparison)
    - RK4 (non-symplectic, for comparison)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Optional


def leapfrog_step(
    model: nn.Module,
    q: torch.Tensor,
    p: torch.Tensor,
    dt: float,
    external_input: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    One step of leapfrog / Stormer-Verlet integration.

    This is a symplectic integrator:
        p_{n+1/2} = p_n - (dt/2) * dp/dt(q_n, p_n)
        q_{n+1}   = q_n + dt * dq/dt(q_n, p_{n+1/2})
        p_{n+1}   = p_{n+1/2} - (dt/2) * dp/dt(q_{n+1}, p_{n+1/2})

    Note: We negate dp/dt because dp/dt = -dH/dq (from Hamilton's equations),
    so the force term is already included in the model's time_derivative.

    Args:
        model: HNN or DissipativeHNN with time_derivative(q, p) method.
        q: Current generalized coordinates, shape (batch, coord_dim).
        p: Current conjugate momenta, shape (batch, coord_dim).
        dt: Time step size.
        external_input: Optional external input for Port-Hamiltonian models.

    Returns:
        q_new: Updated coordinates.
        p_new: Updated momenta.
    """
    # Half-step momentum update
    if external_input is not None:
        _, dp_dt = model.time_derivative(q, p, external_input)
    else:
        _, dp_dt = model.time_derivative(q, p)
    p_half = p + 0.5 * dt * dp_dt

    # Full-step position update
    if external_input is not None:
        dq_dt, _ = model.time_derivative(q, p_half, external_input)
    else:
        dq_dt, _ = model.time_derivative(q, p_half)
    q_new = q + dt * dq_dt

    # Half-step momentum update
    if external_input is not None:
        _, dp_dt = model.time_derivative(q_new, p_half, external_input)
    else:
        _, dp_dt = model.time_derivative(q_new, p_half)
    p_new = p_half + 0.5 * dt * dp_dt

    return q_new, p_new


def euler_step(
    model: nn.Module,
    q: torch.Tensor,
    p: torch.Tensor,
    dt: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    One step of forward Euler integration (non-symplectic, for comparison).

    Args:
        model: HNN with time_derivative(q, p) method.
        q: Current coordinates.
        p: Current momenta.
        dt: Time step.

    Returns:
        q_new, p_new: Updated state.
    """
    dq_dt, dp_dt = model.time_derivative(q, p)
    q_new = q + dt * dq_dt
    p_new = p + dt * dp_dt
    return q_new, p_new


def rk4_step(
    model: nn.Module,
    q: torch.Tensor,
    p: torch.Tensor,
    dt: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    One step of 4th-order Runge-Kutta integration (non-symplectic, for comparison).

    Higher accuracy per step than Euler, but still not symplectic
    (energy drifts over long horizons).

    Args:
        model: HNN with time_derivative(q, p) method.
        q: Current coordinates.
        p: Current momenta.
        dt: Time step.

    Returns:
        q_new, p_new: Updated state.
    """
    # k1
    dq1, dp1 = model.time_derivative(q, p)

    # k2
    dq2, dp2 = model.time_derivative(
        q + 0.5 * dt * dq1, p + 0.5 * dt * dp1
    )

    # k3
    dq3, dp3 = model.time_derivative(
        q + 0.5 * dt * dq2, p + 0.5 * dt * dp2
    )

    # k4
    dq4, dp4 = model.time_derivative(
        q + dt * dq3, p + dt * dp3
    )

    q_new = q + (dt / 6.0) * (dq1 + 2 * dq2 + 2 * dq3 + dq4)
    p_new = p + (dt / 6.0) * (dp1 + 2 * dp2 + 2 * dp3 + dp4)

    return q_new, p_new


def integrate_trajectory(
    model: nn.Module,
    q0: torch.Tensor,
    p0: torch.Tensor,
    dt: float,
    n_steps: int,
    method: str = "leapfrog",
    external_inputs: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Integrate a trajectory through phase space.

    Args:
        model: HNN model with time_derivative(q, p) method.
        q0: Initial coordinates, shape (batch, coord_dim).
        p0: Initial momenta, shape (batch, coord_dim).
        dt: Integration time step.
        n_steps: Number of integration steps.
        method: Integration method ('leapfrog', 'euler', 'rk4').
        external_inputs: Optional tensor of external inputs over time,
                        shape (n_steps, batch, external_dim).

    Returns:
        traj_q: Trajectory of q, shape (n_steps+1, batch, coord_dim).
        traj_p: Trajectory of p, shape (n_steps+1, batch, coord_dim).
    """
    step_fn = {
        "leapfrog": leapfrog_step,
        "euler": euler_step,
        "rk4": rk4_step,
    }

    if method not in step_fn:
        raise ValueError(f"Unknown method: {method}. Use: {list(step_fn.keys())}")

    traj_q: List[torch.Tensor] = [q0.detach()]
    traj_p: List[torch.Tensor] = [p0.detach()]

    q, p = q0.clone(), p0.clone()

    for step in range(n_steps):
        if method == "leapfrog" and external_inputs is not None:
            u = external_inputs[step] if step < len(external_inputs) else None
            q, p = leapfrog_step(model, q, p, dt, u)
        else:
            q, p = step_fn[method](model, q, p, dt)

        traj_q.append(q.detach())
        traj_p.append(p.detach())

    return torch.stack(traj_q), torch.stack(traj_p)


def compute_energy_along_trajectory(
    model: nn.Module,
    traj_q: torch.Tensor,
    traj_p: torch.Tensor,
) -> np.ndarray:
    """
    Compute the Hamiltonian (energy) along a trajectory.

    Args:
        model: HNN model.
        traj_q: Trajectory of q, shape (n_steps+1, batch, coord_dim).
        traj_p: Trajectory of p, shape (n_steps+1, batch, coord_dim).

    Returns:
        energies: Array of energies, shape (n_steps+1, batch).
    """
    energies = []
    with torch.no_grad():
        for i in range(len(traj_q)):
            H = model.hamiltonian(traj_q[i], traj_p[i])
            energies.append(H.squeeze(-1).numpy())
    return np.array(energies)


def compare_integrators(
    model: nn.Module,
    q0: torch.Tensor,
    p0: torch.Tensor,
    dt: float,
    n_steps: int,
) -> dict:
    """
    Compare different integration methods on energy conservation.

    Args:
        model: HNN model.
        q0, p0: Initial state.
        dt: Time step.
        n_steps: Number of steps.

    Returns:
        Dictionary with trajectory and energy data for each method.
    """
    results = {}

    for method in ["leapfrog", "euler", "rk4"]:
        traj_q, traj_p = integrate_trajectory(model, q0, p0, dt, n_steps, method)
        energies = compute_energy_along_trajectory(model, traj_q, traj_p)

        results[method] = {
            "traj_q": traj_q.numpy(),
            "traj_p": traj_p.numpy(),
            "energies": energies,
            "energy_drift": float(energies[-1].mean() - energies[0].mean()),
            "energy_std": float(energies.std()),
        }

    return results


if __name__ == "__main__":
    from hamiltonian_nn import HamiltonianNN

    print("Comparing integrators on trained HNN...\n")

    # Create a simple harmonic oscillator HNN (pretrained-like)
    model = HamiltonianNN(input_dim=2, hidden_dim=64, separable=True)

    # Quick training on harmonic oscillator
    N = 500
    t = np.linspace(0, 4 * np.pi, N)
    q_true = np.cos(t).reshape(-1, 1)
    p_true = -np.sin(t).reshape(-1, 1)
    dq_dt_true = -np.sin(t).reshape(-1, 1)
    dp_dt_true = -np.cos(t).reshape(-1, 1)

    q_t = torch.FloatTensor(q_true)
    p_t = torch.FloatTensor(p_true)
    dq_target = torch.FloatTensor(dq_dt_true)
    dp_target = torch.FloatTensor(dp_dt_true)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(300):
        optimizer.zero_grad()
        dq_pred, dp_pred = model.time_derivative(q_t, p_t)
        loss = ((dq_pred - dq_target) ** 2).mean() + (
            (dp_pred - dp_target) ** 2
        ).mean()
        loss.backward()
        optimizer.step()

    print(f"Training loss: {loss.item():.6f}\n")

    # Compare integrators
    q0 = torch.FloatTensor([[1.0]])
    p0 = torch.FloatTensor([[0.0]])
    dt = 0.05
    n_steps = 200

    results = compare_integrators(model, q0, p0, dt, n_steps)

    print(f"{'Method':<12} {'Energy Drift':>14} {'Energy Std':>12}")
    print("-" * 40)
    for method, data in results.items():
        print(
            f"{method:<12} {data['energy_drift']:>14.8f} {data['energy_std']:>12.8f}"
        )

    print("\nLeapfrog should have minimal drift and bounded oscillation.")
    print("Euler typically shows large drift.")
    print("RK4 shows moderate drift but better than Euler.")
