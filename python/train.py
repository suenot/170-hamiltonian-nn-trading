"""
Training pipeline for Hamiltonian Neural Networks.

Supports:
  - Standard HNN (energy-conserving)
  - Dissipative HNN (with market friction)
  - Port-Hamiltonian NN (with external inputs)

Usage:
    python train.py --model hnn --epochs 500 --lr 1e-3
    python train.py --model dissipative --epochs 500
    python train.py --model port --epochs 500 --external-dim 3
"""

import argparse
import os
import sys
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from hamiltonian_nn import HamiltonianNN, MultiScaleHNN, compute_hnn_loss
from dissipative_hnn import (
    DissipativeHNN,
    PortHamiltonianNN,
    compute_dissipative_loss,
)
from data_loader import (
    fetch_bybit_extended,
    fetch_yahoo_data,
    construct_phase_space,
    normalize_phase_space,
    train_test_split_sequential,
)
from symplectic_integrator import integrate_trajectory, compute_energy_along_trajectory


def create_model(
    model_type: str,
    coord_dim: int,
    hidden_dim: int = 64,
    num_layers: int = 3,
    separable: bool = False,
    external_dim: int = 0,
) -> nn.Module:
    """
    Create an HNN model based on type.

    Args:
        model_type: One of "hnn", "dissipative", "port", "multiscale".
        coord_dim: Dimension of generalized coordinates.
        hidden_dim: Width of hidden layers.
        num_layers: Number of hidden layers.
        separable: If True and model_type="hnn", use separable Hamiltonian.
        external_dim: Dimension of external input (for port model).

    Returns:
        PyTorch model.
    """
    input_dim = 2 * coord_dim

    if model_type == "hnn":
        return HamiltonianNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            separable=separable,
        )
    elif model_type == "dissipative":
        return DissipativeHNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
    elif model_type == "port":
        return PortHamiltonianNN(
            coord_dim=coord_dim,
            external_dim=external_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
    elif model_type == "multiscale":
        return MultiScaleHNN(
            coord_dim=coord_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    model_type: str,
    energy_reg: float = 0.0,
    dissipation_reg: float = 0.01,
    grad_clip: float = 1.0,
) -> dict:
    """
    Train for one epoch.

    Returns:
        Dictionary of average metrics.
    """
    model.train()
    total_metrics: dict = {}
    n_batches = 0

    for batch in dataloader:
        q, p, dq_target, dp_target = batch
        optimizer.zero_grad()

        if model_type in ("hnn", "multiscale"):
            loss, metrics = compute_hnn_loss(
                model, q, p, dq_target, dp_target, energy_reg=energy_reg
            )
        elif model_type in ("dissipative", "port"):
            loss, metrics = compute_dissipative_loss(
                model, q, p, dq_target, dp_target, dissipation_reg=dissipation_reg
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0) + v
        n_batches += 1

    # Average metrics
    return {k: v / n_batches for k, v in total_metrics.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    model_type: str,
) -> dict:
    """
    Evaluate model on validation data.

    Returns:
        Dictionary of average metrics.
    """
    model.eval()
    total_metrics: dict = {}
    n_batches = 0

    for batch in dataloader:
        q, p, dq_target, dp_target = batch

        # We need gradients for autograd in time_derivative
        q = q.requires_grad_(True)
        p = p.requires_grad_(True)

        if model_type in ("hnn", "multiscale"):
            dq_pred, dp_pred = model.time_derivative(q, p)
            loss_q = ((dq_pred - dq_target) ** 2).mean()
            loss_p = ((dp_pred - dp_target) ** 2).mean()
            metrics = {
                "loss_q": loss_q.item(),
                "loss_p": loss_p.item(),
                "loss_total": (loss_q + loss_p).item(),
            }
        else:
            dq_pred, dp_pred = model.time_derivative(q, p)
            loss_q = ((dq_pred - dq_target) ** 2).mean()
            loss_p = ((dp_pred - dp_target) ** 2).mean()
            metrics = {
                "loss_q": loss_q.item(),
                "loss_p": loss_p.item(),
                "loss_total": (loss_q + loss_p).item(),
            }

        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0) + v
        n_batches += 1

    return {k: v / n_batches for k, v in total_metrics.items()}


def evaluate_energy_conservation(
    model: nn.Module,
    q_test: torch.Tensor,
    p_test: torch.Tensor,
    dt: float = 0.1,
    n_steps: int = 100,
) -> dict:
    """
    Evaluate energy conservation over a trajectory.

    Returns:
        Dictionary with energy statistics.
    """
    model.eval()
    # Use first test sample as initial condition
    q0 = q_test[:1]
    p0 = p_test[:1]

    traj_q, traj_p = integrate_trajectory(
        model, q0, p0, dt, n_steps, method="leapfrog"
    )
    energies = compute_energy_along_trajectory(model, traj_q, traj_p)

    return {
        "energy_mean": float(energies.mean()),
        "energy_std": float(energies.std()),
        "energy_drift": float(energies[-1].mean() - energies[0].mean()),
        "energy_relative_drift": float(
            (energies[-1].mean() - energies[0].mean())
            / (abs(energies[0].mean()) + 1e-10)
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Train Hamiltonian Neural Network")

    # Data arguments
    parser.add_argument(
        "--source", type=str, default="bybit", choices=["bybit", "yahoo"]
    )
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--interval", type=str, default="5")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--ma-window", type=int, default=20)

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="hnn",
        choices=["hnn", "dissipative", "port", "multiscale"],
    )
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--separable", action="store_true")
    parser.add_argument("--external-dim", type=int, default=3)

    # Training arguments
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--energy-reg", type=float, default=0.01)
    parser.add_argument("--dissipation-reg", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "step", "none"])

    # Output
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--save-model", type=str, default="saved_model.pt")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Fetch Data ----
    print(f"Fetching data: {args.source}/{args.symbol}...")
    if args.source == "bybit":
        df = fetch_bybit_extended(
            symbol=args.symbol,
            interval=args.interval,
            total_candles=args.limit,
        )
    else:
        df = fetch_yahoo_data(symbol=args.symbol, period="2y", interval="1d")

    print(f"Loaded {len(df)} candles")

    if len(df) < 100:
        print("ERROR: Not enough data. Need at least 100 candles.")
        sys.exit(1)

    # ---- Construct Phase Space ----
    print("Constructing phase space...")
    q, p, dq_dt, dp_dt = construct_phase_space(
        df, ma_window=args.ma_window, momentum_type="returns"
    )
    q_norm, p_norm, dq_norm, dp_norm, stats = normalize_phase_space(
        q, p, dq_dt, dp_dt
    )

    coord_dim = q_norm.shape[1]
    print(f"Phase space: coord_dim={coord_dim}, samples={len(q_norm)}")

    # ---- Train/Test Split ----
    splits = train_test_split_sequential(q_norm, p_norm, dq_norm, dp_norm)
    q_train, p_train, dq_train, dp_train = splits[:4]
    q_test, p_test, dq_test, dp_test = splits[4:]
    print(f"Train: {len(q_train)}, Test: {len(q_test)}")

    # ---- Create DataLoaders ----
    train_dataset = TensorDataset(
        torch.FloatTensor(q_train),
        torch.FloatTensor(p_train),
        torch.FloatTensor(dq_train),
        torch.FloatTensor(dp_train),
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(q_test),
        torch.FloatTensor(p_test),
        torch.FloatTensor(dq_test),
        torch.FloatTensor(dp_test),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # ---- Create Model ----
    model = create_model(
        model_type=args.model,
        coord_dim=coord_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        separable=args.separable,
        external_dim=args.external_dim,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {args.model} ({n_params} parameters)")

    # ---- Optimizer and Scheduler ----
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.epochs // 3, gamma=0.5
        )

    # ---- Training Loop ----
    print(f"\nTraining for {args.epochs} epochs...")
    best_val_loss = float("inf")
    history: list = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            args.model,
            energy_reg=args.energy_reg,
            dissipation_reg=args.dissipation_reg,
            grad_clip=args.grad_clip,
        )

        # Evaluate
        val_metrics = evaluate(model, test_loader, args.model)

        if scheduler is not None:
            scheduler.step()

        elapsed = time.time() - t0

        entry = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
            "time": elapsed,
        }
        history.append(entry)

        # Save best model
        if val_metrics["loss_total"] < best_val_loss:
            best_val_loss = val_metrics["loss_total"]
            save_path = os.path.join(args.output_dir, args.save_model)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_type": args.model,
                    "coord_dim": coord_dim,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "separable": args.separable,
                    "stats": stats,
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                },
                save_path,
            )

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d}/{args.epochs}: "
                f"train_loss={train_metrics['loss_total']:.6f}, "
                f"val_loss={val_metrics['loss_total']:.6f}, "
                f"lr={optimizer.param_groups[0]['lr']:.6f}, "
                f"time={elapsed:.2f}s"
            )

    # ---- Final Evaluation ----
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")

    # Energy conservation check
    print("\nEnergy conservation evaluation:")
    q_test_t = torch.FloatTensor(q_test)
    p_test_t = torch.FloatTensor(p_test)
    energy_stats = evaluate_energy_conservation(model, q_test_t, p_test_t)
    for k, v in energy_stats.items():
        print(f"  {k}: {v:.8f}")

    # ---- Save History ----
    history_path = os.path.join(args.output_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to {history_path}")

    # Save stats for inference
    stats_path = os.path.join(args.output_dir, "normalization_stats.json")
    stats_serializable = {k: v.tolist() for k, v in stats.items()}
    with open(stats_path, "w") as f:
        json.dump(stats_serializable, f, indent=2)

    print(f"Model saved to {os.path.join(args.output_dir, args.save_model)}")


if __name__ == "__main__":
    main()
