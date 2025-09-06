"""
Hamiltonian Neural Network (HNN) for financial market modeling.

The HNN learns a scalar Hamiltonian function H_theta(q, p) and derives
the system dynamics via Hamilton's equations using automatic differentiation:
    dq/dt =  dH/dp
    dp/dt = -dH/dq

This ensures energy conservation by construction, providing stable
long-horizon predictions suitable for trading applications.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List


class HamiltonianNN(nn.Module):
    """
    Hamiltonian Neural Network.

    Learns the Hamiltonian H(q, p) as a scalar function.
    Dynamics are derived via Hamilton's equations using autograd.

    Args:
        input_dim: Total dimension of (q, p) concatenated.
                   For 1D system, q and p are each 1D, so input_dim=2.
                   For N-asset system, input_dim = 2*N.
        hidden_dim: Width of hidden layers.
        num_layers: Number of hidden layers.
        activation: Activation function (must be smooth for Hamilton's equations).
        separable: If True, learn T(p) + V(q) separately (simpler, more constrained).
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 3,
        activation: str = "tanh",
        separable: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.coord_dim = input_dim // 2
        self.separable = separable

        act_fn = self._get_activation(activation)

        if separable:
            # Separable Hamiltonian: H(q, p) = T(p) + V(q)
            self.T_net = self._build_mlp(self.coord_dim, hidden_dim, num_layers, act_fn)
            self.V_net = self._build_mlp(self.coord_dim, hidden_dim, num_layers, act_fn)
        else:
            # General (non-separable) Hamiltonian: H(q, p) = f(q, p)
            self.H_net = self._build_mlp(input_dim, hidden_dim, num_layers, act_fn)

    @staticmethod
    def _get_activation(name: str) -> nn.Module:
        """Get smooth activation function."""
        activations = {
            "tanh": nn.Tanh(),
            "softplus": nn.Softplus(),
            "sigmoid": nn.Sigmoid(),
            "silu": nn.SiLU(),
            "gelu": nn.GELU(),
        }
        if name not in activations:
            raise ValueError(
                f"Activation '{name}' not supported. Use smooth activations: {list(activations.keys())}"
            )
        return activations[name]

    @staticmethod
    def _build_mlp(
        input_dim: int, hidden_dim: int, num_layers: int, act_fn: nn.Module
    ) -> nn.Sequential:
        """Build MLP with smooth activations and scalar output."""
        layers: List[nn.Module] = [nn.Linear(input_dim, hidden_dim), act_fn]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), act_fn])
        layers.append(nn.Linear(hidden_dim, 1))
        return nn.Sequential(*layers)

    def hamiltonian(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        Compute the Hamiltonian H(q, p).

        Args:
            q: Generalized coordinates, shape (batch, coord_dim)
            p: Conjugate momenta, shape (batch, coord_dim)

        Returns:
            H: Hamiltonian (scalar energy), shape (batch, 1)
        """
        if self.separable:
            T = self.T_net(p)  # Kinetic energy
            V = self.V_net(q)  # Potential energy
            return T + V
        else:
            x = torch.cat([q, p], dim=-1)
            return self.H_net(x)

    def forward(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Alias for hamiltonian()."""
        return self.hamiltonian(q, p)

    def time_derivative(
        self, q: torch.Tensor, p: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute time derivatives via Hamilton's equations:
            dq/dt =  dH/dp
            dp/dt = -dH/dq

        Args:
            q: Generalized coordinates, shape (batch, coord_dim)
            p: Conjugate momenta, shape (batch, coord_dim)

        Returns:
            dq_dt: Time derivative of q, shape (batch, coord_dim)
            dp_dt: Time derivative of p, shape (batch, coord_dim)
        """
        q = q.detach().requires_grad_(True)
        p = p.detach().requires_grad_(True)

        H = self.hamiltonian(q, p)

        dH_dq, dH_dp = torch.autograd.grad(
            H.sum(), [q, p], create_graph=True, retain_graph=True
        )

        dq_dt = dH_dp  # Hamilton's first equation
        dp_dt = -dH_dq  # Hamilton's second equation

        return dq_dt, dp_dt

    def energy(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Compute energy (Hamiltonian value) without gradients."""
        with torch.no_grad():
            return self.hamiltonian(q, p)

    def energy_components(
        self, q: torch.Tensor, p: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        """
        Decompose energy into kinetic and potential (only for separable HNN).

        Returns:
            (T, V, H) if separable, else (None, None, H)
        """
        if self.separable:
            with torch.no_grad():
                T = self.T_net(p)
                V = self.V_net(q)
                H = T + V
            return T, V, H
        else:
            H = self.energy(q, p)
            return None, None, H


class MultiScaleHNN(nn.Module):
    """
    Multi-scale Hamiltonian Neural Network.

    Learns Hamiltonians at multiple time scales and combines them.
    Useful for capturing both short-term microstructure dynamics
    and longer-term mean reversion.

    Args:
        coord_dim: Dimension of generalized coordinates.
        scales: List of time scales (in number of steps).
        hidden_dim: Width of hidden layers for each scale.
        num_layers: Number of hidden layers for each scale.
    """

    def __init__(
        self,
        coord_dim: int = 1,
        scales: Optional[List[int]] = None,
        hidden_dim: int = 32,
        num_layers: int = 2,
    ):
        super().__init__()
        if scales is None:
            scales = [1, 5, 20]
        self.scales = scales
        self.coord_dim = coord_dim

        self.hnns = nn.ModuleList([
            HamiltonianNN(
                input_dim=2 * coord_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                separable=True,
            )
            for _ in scales
        ])

        # Learnable scale weights
        self.scale_weights = nn.Parameter(torch.ones(len(scales)) / len(scales))

    def hamiltonian(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Combined multi-scale Hamiltonian."""
        weights = torch.softmax(self.scale_weights, dim=0)
        H = torch.zeros(q.shape[0], 1, device=q.device)
        for w, hnn in zip(weights, self.hnns):
            H = H + w * hnn.hamiltonian(q, p)
        return H

    def time_derivative(
        self, q: torch.Tensor, p: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Multi-scale time derivatives."""
        q = q.detach().requires_grad_(True)
        p = p.detach().requires_grad_(True)

        H = self.hamiltonian(q, p)
        dH_dq, dH_dp = torch.autograd.grad(
            H.sum(), [q, p], create_graph=True, retain_graph=True
        )

        return dH_dp, -dH_dq

    def forward(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return self.hamiltonian(q, p)


def compute_hnn_loss(
    model: nn.Module,
    q: torch.Tensor,
    p: torch.Tensor,
    dq_dt_target: torch.Tensor,
    dp_dt_target: torch.Tensor,
    energy_reg: float = 0.0,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute HNN training loss.

    Loss = MSE(predicted dq/dt, target dq/dt) + MSE(predicted dp/dt, target dp/dt)
         + energy_reg * Var(H)  (optional: penalize energy variation for stability)

    Args:
        model: HNN model.
        q: Generalized coordinates.
        p: Conjugate momenta.
        dq_dt_target: Target dq/dt.
        dp_dt_target: Target dp/dt.
        energy_reg: Weight for energy variance regularization.

    Returns:
        loss: Total loss scalar.
        metrics: Dictionary of individual loss components.
    """
    dq_dt_pred, dp_dt_pred = model.time_derivative(q, p)

    loss_q = ((dq_dt_pred - dq_dt_target) ** 2).mean()
    loss_p = ((dp_dt_pred - dp_dt_target) ** 2).mean()
    loss = loss_q + loss_p

    metrics = {
        "loss_q": loss_q.item(),
        "loss_p": loss_p.item(),
        "loss_total": loss.item(),
    }

    if energy_reg > 0:
        H = model.hamiltonian(q, p)
        energy_var = H.var()
        loss = loss + energy_reg * energy_var
        metrics["energy_var"] = energy_var.item()

    return loss, metrics


if __name__ == "__main__":
    # Quick test: learn a simple harmonic oscillator
    print("Testing HNN on harmonic oscillator...")

    # Generate synthetic data: H(q, p) = 0.5 * (q^2 + p^2)
    N = 1000
    t = np.linspace(0, 10 * np.pi, N)
    q_true = np.cos(t).reshape(-1, 1)
    p_true = -np.sin(t).reshape(-1, 1)
    dq_dt_true = -np.sin(t).reshape(-1, 1)
    dp_dt_true = -np.cos(t).reshape(-1, 1)

    q_t = torch.FloatTensor(q_true)
    p_t = torch.FloatTensor(p_true)
    dq_target = torch.FloatTensor(dq_dt_true)
    dp_target = torch.FloatTensor(dp_dt_true)

    # Train HNN
    model = HamiltonianNN(input_dim=2, hidden_dim=64, num_layers=3, separable=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(200):
        optimizer.zero_grad()
        loss, metrics = compute_hnn_loss(model, q_t, p_t, dq_target, dp_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if epoch % 50 == 0:
            H = model.energy(q_t, p_t)
            print(
                f"  Epoch {epoch}: loss={metrics['loss_total']:.6f}, "
                f"H_mean={H.mean().item():.4f}, H_std={H.std().item():.4f}"
            )

    # Check energy conservation
    H_final = model.energy(q_t, p_t)
    print(f"\nEnergy conservation check:")
    print(f"  H mean: {H_final.mean().item():.4f}")
    print(f"  H std:  {H_final.std().item():.4f}")
    print(f"  H range: [{H_final.min().item():.4f}, {H_final.max().item():.4f}]")
    print("  (Ideally H_std should be close to 0, meaning energy is conserved)")
