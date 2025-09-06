"""
Visualization tools for Hamiltonian Neural Networks.

Generates:
  - Phase portraits (q vs p trajectories)
  - Energy conservation plots
  - Vector field plots
  - Predicted vs actual trajectory comparisons
  - Energy decomposition (kinetic vs potential)
"""

import argparse
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from hamiltonian_nn import HamiltonianNN
from dissipative_hnn import DissipativeHNN
from symplectic_integrator import (
    integrate_trajectory,
    compute_energy_along_trajectory,
    compare_integrators,
)
from train import create_model


def plot_phase_portrait(
    q: np.ndarray,
    p: np.ndarray,
    title: str = "Phase Portrait",
    color_by_time: bool = True,
    ax: plt.Axes = None,
    save_path: str = None,
):
    """
    Plot phase portrait (q vs p).

    Args:
        q: Coordinates, shape (N,) or (N, 1).
        p: Momenta, shape (N,) or (N, 1).
        title: Plot title.
        color_by_time: If True, color points by time index.
        ax: Matplotlib axes (optional).
        save_path: Path to save figure (optional).
    """
    q = q.squeeze()
    p = p.squeeze()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    if color_by_time:
        colors = np.arange(len(q))
        sc = ax.scatter(q, p, c=colors, cmap="viridis", s=3, alpha=0.6)
        plt.colorbar(sc, ax=ax, label="Time step")
    else:
        ax.plot(q, p, "b-", alpha=0.5, linewidth=0.5)
        ax.scatter(q[0], p[0], c="green", s=100, zorder=5, label="Start", marker="o")
        ax.scatter(q[-1], p[-1], c="red", s=100, zorder=5, label="End", marker="x")
        ax.legend()

    ax.set_xlabel("q (price deviation)", fontsize=12)
    ax.set_ylabel("p (momentum)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")


def plot_vector_field(
    model: torch.nn.Module,
    q_range: tuple = (-2, 2),
    p_range: tuple = (-2, 2),
    n_grid: int = 20,
    title: str = "Hamiltonian Vector Field",
    save_path: str = None,
):
    """
    Plot the vector field (dq/dt, dp/dt) learned by the HNN.

    Args:
        model: Trained HNN model.
        q_range: Range of q values.
        p_range: Range of p values.
        n_grid: Number of grid points per axis.
        title: Plot title.
        save_path: Save path.
    """
    model.eval()

    q_vals = np.linspace(q_range[0], q_range[1], n_grid)
    p_vals = np.linspace(p_range[0], p_range[1], n_grid)
    Q, P = np.meshgrid(q_vals, p_vals)

    q_flat = torch.FloatTensor(Q.flatten()).unsqueeze(-1)
    p_flat = torch.FloatTensor(P.flatten()).unsqueeze(-1)

    with torch.enable_grad():
        dq_dt, dp_dt = model.time_derivative(q_flat, p_flat)

    DQ = dq_dt.detach().numpy().reshape(Q.shape)
    DP = dp_dt.detach().numpy().reshape(P.shape)

    # Speed for coloring
    speed = np.sqrt(DQ**2 + DP**2)

    fig, ax = plt.subplots(figsize=(10, 8))
    strm = ax.streamplot(
        Q, P, DQ, DP,
        color=speed,
        cmap="coolwarm",
        linewidth=1.5,
        density=1.5,
        arrowsize=1.5,
    )
    plt.colorbar(strm.lines, ax=ax, label="Speed |dz/dt|")

    ax.set_xlabel("q (price deviation)", fontsize=12)
    ax.set_ylabel("p (momentum)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")


def plot_energy_surface(
    model: torch.nn.Module,
    q_range: tuple = (-2, 2),
    p_range: tuple = (-2, 2),
    n_grid: int = 50,
    title: str = "Hamiltonian Energy Surface",
    save_path: str = None,
):
    """
    Plot the learned Hamiltonian H(q, p) as a contour plot.

    Args:
        model: Trained HNN model.
        q_range, p_range: Ranges for grid.
        n_grid: Grid resolution.
        title: Plot title.
        save_path: Save path.
    """
    model.eval()

    q_vals = np.linspace(q_range[0], q_range[1], n_grid)
    p_vals = np.linspace(p_range[0], p_range[1], n_grid)
    Q, P = np.meshgrid(q_vals, p_vals)

    q_flat = torch.FloatTensor(Q.flatten()).unsqueeze(-1)
    p_flat = torch.FloatTensor(P.flatten()).unsqueeze(-1)

    with torch.no_grad():
        H = model.hamiltonian(q_flat, p_flat)

    H_grid = H.numpy().reshape(Q.shape)

    fig, ax = plt.subplots(figsize=(10, 8))
    cf = ax.contourf(Q, P, H_grid, levels=30, cmap="RdYlBu_r")
    ax.contour(Q, P, H_grid, levels=15, colors="black", linewidths=0.5, alpha=0.5)
    plt.colorbar(cf, ax=ax, label="H(q, p)")

    ax.set_xlabel("q (price deviation)", fontsize=12)
    ax.set_ylabel("p (momentum)", fontsize=12)
    ax.set_title(title, fontsize=14)

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")


def plot_energy_conservation(
    energies_dict: dict,
    title: str = "Energy Conservation Comparison",
    save_path: str = None,
):
    """
    Plot energy over time for different integration methods.

    Args:
        energies_dict: {method_name: energies_array}.
        title: Plot title.
        save_path: Save path.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Absolute energy
    ax1 = axes[0]
    for method, energies in energies_dict.items():
        ax1.plot(energies.squeeze(), label=method, linewidth=1.5)
    ax1.set_xlabel("Integration step", fontsize=12)
    ax1.set_ylabel("H(q, p)", fontsize=12)
    ax1.set_title("Energy over time", fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Relative energy error
    ax2 = axes[1]
    for method, energies in energies_dict.items():
        e = energies.squeeze()
        relative_error = (e - e[0]) / (abs(e[0]) + 1e-10)
        ax2.plot(relative_error, label=method, linewidth=1.5)
    ax2.set_xlabel("Integration step", fontsize=12)
    ax2.set_ylabel("(H - H_0) / |H_0|", fontsize=12)
    ax2.set_title("Relative energy error", fontsize=13)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")


def plot_trajectory_comparison(
    q_actual: np.ndarray,
    p_actual: np.ndarray,
    q_predicted: np.ndarray,
    p_predicted: np.ndarray,
    title: str = "Actual vs Predicted Trajectory",
    save_path: str = None,
):
    """
    Compare actual and predicted trajectories in phase space and time domain.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Phase portrait
    ax1 = axes[0]
    ax1.plot(
        q_actual.squeeze(),
        p_actual.squeeze(),
        "b-",
        alpha=0.7,
        linewidth=1,
        label="Actual",
    )
    ax1.plot(
        q_predicted.squeeze(),
        p_predicted.squeeze(),
        "r--",
        alpha=0.7,
        linewidth=1,
        label="Predicted",
    )
    ax1.set_xlabel("q", fontsize=12)
    ax1.set_ylabel("p", fontsize=12)
    ax1.set_title("Phase Space", fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # q over time
    ax2 = axes[1]
    n = min(len(q_actual), len(q_predicted))
    ax2.plot(q_actual[:n].squeeze(), "b-", alpha=0.7, label="Actual q")
    ax2.plot(q_predicted[:n].squeeze(), "r--", alpha=0.7, label="Predicted q")
    ax2.set_xlabel("Time step", fontsize=12)
    ax2.set_ylabel("q (price deviation)", fontsize=12)
    ax2.set_title("Position over time", fontsize=13)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # p over time
    ax3 = axes[2]
    ax3.plot(p_actual[:n].squeeze(), "b-", alpha=0.7, label="Actual p")
    ax3.plot(p_predicted[:n].squeeze(), "r--", alpha=0.7, label="Predicted p")
    ax3.set_xlabel("Time step", fontsize=12)
    ax3.set_ylabel("p (momentum)", fontsize=12)
    ax3.set_title("Momentum over time", fontsize=13)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")


def plot_dissipation_analysis(
    model: DissipativeHNN,
    q_range: tuple = (-2, 2),
    p_range: tuple = (-2, 2),
    n_grid: int = 50,
    save_path: str = None,
):
    """
    Plot dissipation function D(q, p) and energy rate dH/dt.
    """
    model.eval()

    q_vals = np.linspace(q_range[0], q_range[1], n_grid)
    p_vals = np.linspace(p_range[0], p_range[1], n_grid)
    Q, P = np.meshgrid(q_vals, p_vals)

    q_flat = torch.FloatTensor(Q.flatten()).unsqueeze(-1)
    p_flat = torch.FloatTensor(P.flatten()).unsqueeze(-1)

    with torch.no_grad():
        D = model.dissipation(q_flat, p_flat)
    D_grid = D.numpy().reshape(Q.shape)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Dissipation surface
    ax1 = axes[0]
    cf1 = ax1.contourf(Q, P, D_grid, levels=30, cmap="Reds")
    plt.colorbar(cf1, ax=ax1, label="D(q, p)")
    ax1.set_xlabel("q", fontsize=12)
    ax1.set_ylabel("p", fontsize=12)
    ax1.set_title("Dissipation Function D(q, p)", fontsize=13)

    # Energy rate dH/dt
    with torch.enable_grad():
        dH_dt = model.energy_rate(q_flat, p_flat)
    dH_dt_grid = dH_dt.detach().numpy().reshape(Q.shape)

    ax2 = axes[1]
    vmax = max(abs(dH_dt_grid.min()), abs(dH_dt_grid.max()))
    cf2 = ax2.contourf(
        Q, P, dH_dt_grid,
        levels=30,
        cmap="RdBu_r",
        norm=mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax),
    )
    plt.colorbar(cf2, ax=ax2, label="dH/dt")
    ax2.set_xlabel("q", fontsize=12)
    ax2.set_ylabel("p", fontsize=12)
    ax2.set_title("Energy Rate dH/dt (should be <= 0)", fontsize=13)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")


def create_all_visualizations(
    model: torch.nn.Module,
    q_data: np.ndarray,
    p_data: np.ndarray,
    model_type: str,
    output_dir: str = "output/plots",
):
    """Generate all visualization plots."""
    os.makedirs(output_dir, exist_ok=True)

    print("Generating visualizations...")

    # 1. Data phase portrait
    print("  - Phase portrait (data)")
    plot_phase_portrait(
        q_data,
        p_data,
        title="Market Phase Portrait (Data)",
        save_path=os.path.join(output_dir, "phase_portrait_data.png"),
    )

    # 2. Vector field
    q_range = (float(q_data.min()) * 1.2, float(q_data.max()) * 1.2)
    p_range = (float(p_data.min()) * 1.2, float(p_data.max()) * 1.2)

    print("  - Vector field")
    plot_vector_field(
        model,
        q_range=q_range,
        p_range=p_range,
        title="Learned Hamiltonian Vector Field",
        save_path=os.path.join(output_dir, "vector_field.png"),
    )

    # 3. Energy surface
    print("  - Energy surface")
    plot_energy_surface(
        model,
        q_range=q_range,
        p_range=p_range,
        title="Learned Hamiltonian H(q, p)",
        save_path=os.path.join(output_dir, "energy_surface.png"),
    )

    # 4. Integration comparison
    print("  - Integration comparison")
    q0 = torch.FloatTensor(q_data[:1])
    p0 = torch.FloatTensor(p_data[:1])
    results = compare_integrators(model, q0, p0, dt=0.05, n_steps=200)
    energies_dict = {m: r["energies"] for m, r in results.items()}
    plot_energy_conservation(
        energies_dict,
        title="Integrator Comparison: Energy Conservation",
        save_path=os.path.join(output_dir, "energy_conservation.png"),
    )

    # 5. Trajectory prediction
    print("  - Trajectory prediction")
    traj_q, traj_p = integrate_trajectory(
        model, q0, p0, dt=0.1, n_steps=min(200, len(q_data) - 1), method="leapfrog"
    )
    n_compare = min(len(traj_q), len(q_data))
    plot_trajectory_comparison(
        q_data[:n_compare],
        p_data[:n_compare],
        traj_q.numpy()[:n_compare, 0],
        traj_p.numpy()[:n_compare, 0],
        title="Actual vs HNN-Predicted Trajectory",
        save_path=os.path.join(output_dir, "trajectory_comparison.png"),
    )

    # 6. Dissipation analysis (if applicable)
    if model_type == "dissipative" and isinstance(model, DissipativeHNN):
        print("  - Dissipation analysis")
        plot_dissipation_analysis(
            model,
            q_range=q_range,
            p_range=p_range,
            save_path=os.path.join(output_dir, "dissipation_analysis.png"),
        )

    plt.close("all")
    print(f"\nAll plots saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Visualize HNN model")
    parser.add_argument("--model", type=str, required=True, help="Path to saved model")
    parser.add_argument(
        "--data", type=str, default=None, help="Path to phase space data (.npz)"
    )
    parser.add_argument("--output-dir", type=str, default="output/plots")
    args = parser.parse_args()

    # Load model
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    model_type = checkpoint["model_type"]
    model = create_model(
        model_type=model_type,
        coord_dim=checkpoint["coord_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_layers=checkpoint["num_layers"],
        separable=checkpoint.get("separable", False),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded {model_type} model from {args.model}")

    # Load or generate data
    if args.data and os.path.exists(args.data):
        data = np.load(args.data)
        q_data = data["q"]
        p_data = data["p"]
    else:
        print("No data file provided. Generating synthetic data for visualization.")
        N = 500
        t = np.linspace(0, 8 * np.pi, N)
        q_data = np.cos(t).reshape(-1, 1)
        p_data = -np.sin(t).reshape(-1, 1)

    create_all_visualizations(model, q_data, p_data, model_type, args.output_dir)


if __name__ == "__main__":
    main()
