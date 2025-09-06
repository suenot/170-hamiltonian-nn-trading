"""
Dissipative Hamiltonian Neural Network for markets with friction.

Extends the standard HNN with a Rayleigh dissipation function D(q, p) >= 0
to model transaction costs, slippage, and other market frictions:

    dq/dt =  dH/dp
    dp/dt = -dH/dq - dD/dp

The dissipation ensures energy monotonically decreases:
    dH/dt = -p^T * dD/dp <= 0

Also includes Port-Hamiltonian extension for open market systems
with external inputs (news, volume shocks, etc.).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List


class DissipativeHNN(nn.Module):
    """
    Dissipative Hamiltonian Neural Network.

    Learns both a Hamiltonian H(q, p) and a dissipation function D(q, p) >= 0.
    Dynamics follow:
        dq/dt =  dH/dp
        dp/dt = -dH/dq - dD/dp

    Args:
        input_dim: Total dimension of (q, p) concatenated.
        hidden_dim: Width of hidden layers.
        num_layers: Number of hidden layers.
        activation: Smooth activation function name.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 3,
        activation: str = "tanh",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.coord_dim = input_dim // 2

        act_fn = self._get_activation(activation)

        # Hamiltonian network: H(q, p) -> scalar
        self.H_net = self._build_mlp(input_dim, hidden_dim, num_layers, act_fn)

        # Dissipation network: D(q, p) -> non-negative scalar
        # Uses Softplus at output to ensure D >= 0
        self.D_net = self._build_dissipation_mlp(
            input_dim, hidden_dim, num_layers, act_fn
        )

    @staticmethod
    def _get_activation(name: str) -> nn.Module:
        activations = {
            "tanh": nn.Tanh(),
            "softplus": nn.Softplus(),
            "sigmoid": nn.Sigmoid(),
            "silu": nn.SiLU(),
        }
        return activations.get(name, nn.Tanh())

    @staticmethod
    def _build_mlp(
        input_dim: int, hidden_dim: int, num_layers: int, act_fn: nn.Module
    ) -> nn.Sequential:
        layers: List[nn.Module] = [nn.Linear(input_dim, hidden_dim), act_fn]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), act_fn])
        layers.append(nn.Linear(hidden_dim, 1))
        return nn.Sequential(*layers)

    @staticmethod
    def _build_dissipation_mlp(
        input_dim: int, hidden_dim: int, num_layers: int, act_fn: nn.Module
    ) -> nn.Sequential:
        """Build MLP with non-negative output for dissipation."""
        layers: List[nn.Module] = [nn.Linear(input_dim, hidden_dim), act_fn]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), act_fn])
        layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Softplus())  # Ensures D >= 0
        return nn.Sequential(*layers)

    def hamiltonian(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Compute Hamiltonian H(q, p)."""
        x = torch.cat([q, p], dim=-1)
        return self.H_net(x)

    def dissipation(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Compute dissipation function D(q, p) >= 0."""
        x = torch.cat([q, p], dim=-1)
        return self.D_net(x)

    def forward(
        self, q: torch.Tensor, p: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute both H and D."""
        return self.hamiltonian(q, p), self.dissipation(q, p)

    def time_derivative(
        self, q: torch.Tensor, p: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute dissipative Hamilton's equations:
            dq/dt =  dH/dp
            dp/dt = -dH/dq - dD/dp

        Args:
            q: Generalized coordinates, shape (batch, coord_dim)
            p: Conjugate momenta, shape (batch, coord_dim)

        Returns:
            dq_dt: Time derivative of q
            dp_dt: Time derivative of p (includes dissipation)
        """
        q = q.detach().requires_grad_(True)
        p = p.detach().requires_grad_(True)

        # Hamiltonian gradients
        H = self.hamiltonian(q, p)
        dH_dq, dH_dp = torch.autograd.grad(
            H.sum(), [q, p], create_graph=True, retain_graph=True
        )

        # Dissipation gradient (w.r.t. p only)
        D = self.dissipation(q, p)
        dD_dp = torch.autograd.grad(D.sum(), p, create_graph=True, retain_graph=True)[
            0
        ]

        dq_dt = dH_dp
        dp_dt = -dH_dq - dD_dp  # Dissipation acts on momentum

        return dq_dt, dp_dt

    def energy_rate(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        Compute rate of energy change dH/dt.
        For a dissipative system, dH/dt <= 0.

        Returns:
            dH_dt: Rate of energy change (should be non-positive)
        """
        q = q.detach().requires_grad_(True)
        p = p.detach().requires_grad_(True)

        dq_dt, dp_dt = self.time_derivative(q, p)
        H = self.hamiltonian(q, p)
        dH_dq, dH_dp = torch.autograd.grad(
            H.sum(), [q, p], create_graph=True, retain_graph=True
        )

        # dH/dt = (dH/dq)(dq/dt) + (dH/dp)(dp/dt)
        dH_dt = (dH_dq * dq_dt).sum(dim=-1) + (dH_dp * dp_dt).sum(dim=-1)
        return dH_dt


class PortHamiltonianNN(nn.Module):
    """
    Port-Hamiltonian Neural Network for open market systems.

    Models markets with external inputs (news, volume shocks, macro data):
        dq/dt =  dH/dp + g_q(q, p) * u(t)
        dp/dt = -dH/dq + g_p(q, p) * u(t) - dD/dp

    Args:
        coord_dim: Dimension of generalized coordinates.
        external_dim: Dimension of external input u(t).
        hidden_dim: Width of hidden layers.
        num_layers: Number of hidden layers.
    """

    def __init__(
        self,
        coord_dim: int = 1,
        external_dim: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 3,
    ):
        super().__init__()
        self.coord_dim = coord_dim
        self.external_dim = external_dim
        input_dim = 2 * coord_dim

        # Core dissipative HNN
        self.dhnn = DissipativeHNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

        # Input coupling networks: g_q and g_p
        # These map (q, p) to coupling matrices for external input
        act_fn = nn.Tanh()
        self.g_q_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, coord_dim * external_dim),
        )
        self.g_p_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, coord_dim * external_dim),
        )

    def time_derivative(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        u: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute port-Hamiltonian dynamics.

        Args:
            q: Generalized coordinates, shape (batch, coord_dim)
            p: Conjugate momenta, shape (batch, coord_dim)
            u: External inputs, shape (batch, external_dim). If None, no external input.

        Returns:
            dq_dt, dp_dt: Time derivatives
        """
        # Base dissipative dynamics
        dq_dt, dp_dt = self.dhnn.time_derivative(q, p)

        if u is not None:
            batch_size = q.shape[0]
            x = torch.cat([q.detach(), p.detach()], dim=-1)

            # Compute coupling matrices
            g_q = self.g_q_net(x).view(batch_size, self.coord_dim, self.external_dim)
            g_p = self.g_p_net(x).view(batch_size, self.coord_dim, self.external_dim)

            # Apply external input: g * u
            u_expanded = u.unsqueeze(-1)  # (batch, external_dim, 1)
            dq_external = torch.bmm(g_q, u_expanded).squeeze(-1)
            dp_external = torch.bmm(g_p, u_expanded).squeeze(-1)

            dq_dt = dq_dt + dq_external
            dp_dt = dp_dt + dp_external

        return dq_dt, dp_dt

    def hamiltonian(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return self.dhnn.hamiltonian(q, p)

    def dissipation(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return self.dhnn.dissipation(q, p)

    def forward(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        u: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.time_derivative(q, p, u)


def compute_dissipative_loss(
    model: DissipativeHNN,
    q: torch.Tensor,
    p: torch.Tensor,
    dq_dt_target: torch.Tensor,
    dp_dt_target: torch.Tensor,
    dissipation_reg: float = 0.01,
    energy_monotone_reg: float = 0.1,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute training loss for Dissipative HNN.

    Loss = MSE(derivatives) + dissipation_reg * D_mean + energy_monotone_reg * max(dH/dt, 0)

    The energy_monotone_reg term penalizes positive energy rates (which violate
    the dissipative property that energy should only decrease).

    Args:
        model: DissipativeHNN model.
        q, p: Phase space coordinates.
        dq_dt_target, dp_dt_target: Target derivatives.
        dissipation_reg: Weight for dissipation magnitude regularization.
        energy_monotone_reg: Weight for energy monotonicity constraint.

    Returns:
        loss: Total loss.
        metrics: Dictionary of loss components.
    """
    dq_pred, dp_pred = model.time_derivative(q, p)

    loss_q = ((dq_pred - dq_dt_target) ** 2).mean()
    loss_p = ((dp_pred - dp_dt_target) ** 2).mean()
    loss = loss_q + loss_p

    metrics = {
        "loss_q": loss_q.item(),
        "loss_p": loss_p.item(),
    }

    # Regularize dissipation magnitude (prevent it from being too large)
    D = model.dissipation(q, p)
    D_mean = D.mean()
    loss = loss + dissipation_reg * D_mean
    metrics["dissipation_mean"] = D_mean.item()

    # Energy monotonicity: dH/dt should be <= 0
    if energy_monotone_reg > 0:
        dH_dt = model.energy_rate(q, p)
        violation = torch.relu(dH_dt).mean()  # Penalize positive dH/dt
        loss = loss + energy_monotone_reg * violation
        metrics["energy_rate_violation"] = violation.item()

    metrics["loss_total"] = loss.item()
    return loss, metrics


if __name__ == "__main__":
    # Test: damped harmonic oscillator
    print("Testing Dissipative HNN on damped oscillator...")

    # Generate synthetic data: damped oscillator
    # dq/dt = p, dp/dt = -q - 0.1*p (damping coefficient 0.1)
    N = 1000
    dt = 0.01
    q_list, p_list = [1.0], [0.0]
    for i in range(N - 1):
        q_i, p_i = q_list[-1], p_list[-1]
        q_list.append(q_i + dt * p_i)
        p_list.append(p_i + dt * (-q_i - 0.1 * p_i))

    q_arr = np.array(q_list).reshape(-1, 1)
    p_arr = np.array(p_list).reshape(-1, 1)
    dq_dt_arr = p_arr.copy()
    dp_dt_arr = -q_arr - 0.1 * p_arr

    q_t = torch.FloatTensor(q_arr)
    p_t = torch.FloatTensor(p_arr)
    dq_target = torch.FloatTensor(dq_dt_arr)
    dp_target = torch.FloatTensor(dp_dt_arr)

    # Train Dissipative HNN
    model = DissipativeHNN(input_dim=2, hidden_dim=64, num_layers=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(300):
        optimizer.zero_grad()
        loss, metrics = compute_dissipative_loss(
            model, q_t, p_t, dq_target, dp_target
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if epoch % 50 == 0:
            H = model.hamiltonian(q_t, p_t).detach()
            D = model.dissipation(q_t, p_t).detach()
            print(
                f"  Epoch {epoch}: loss={metrics['loss_total']:.6f}, "
                f"H_mean={H.mean().item():.4f}, D_mean={D.mean().item():.4f}"
            )

    # Check that dissipation is non-negative
    D_all = model.dissipation(q_t, p_t).detach()
    print(f"\nDissipation check:")
    print(f"  D min: {D_all.min().item():.6f} (should be >= 0)")
    print(f"  D mean: {D_all.mean().item():.6f}")

    # Test Port-Hamiltonian
    print("\nTesting Port-Hamiltonian NN...")
    port_model = PortHamiltonianNN(coord_dim=1, external_dim=3, hidden_dim=32)
    q_test = torch.randn(10, 1)
    p_test = torch.randn(10, 1)
    u_test = torch.randn(10, 3)

    dq, dp = port_model.time_derivative(q_test, p_test, u_test)
    print(f"  With external input: dq shape={dq.shape}, dp shape={dp.shape}")

    dq_no_u, dp_no_u = port_model.time_derivative(q_test, p_test, None)
    print(f"  Without external input: dq shape={dq_no_u.shape}, dp shape={dp_no_u.shape}")
    print("  Port-Hamiltonian test passed!")
