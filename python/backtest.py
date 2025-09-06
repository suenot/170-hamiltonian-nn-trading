"""
Backtesting framework for HNN-based trading strategy.

The strategy uses the learned Hamiltonian to:
1. Predict future phase-space trajectories via symplectic integration
2. Generate buy/sell signals based on predicted price deviation changes
3. Monitor energy for regime change detection and risk management

Usage:
    python backtest.py --model output/saved_model.pt --initial-capital 100000
"""

import argparse
import os
import json
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from hamiltonian_nn import HamiltonianNN
from dissipative_hnn import DissipativeHNN
from symplectic_integrator import integrate_trajectory
from data_loader import (
    fetch_bybit_extended,
    fetch_yahoo_data,
    construct_phase_space,
    normalize_phase_space,
)
from train import create_model


# =============================================================================
# Data structures
# =============================================================================


@dataclass
class Trade:
    """Record of a single trade."""
    timestamp: int  # Index in data
    side: str  # "BUY" or "SELL"
    price: float
    quantity: float
    signal_strength: float  # Predicted change magnitude
    energy: float  # System energy at trade time
    energy_zscore: float  # Energy z-score (regime detection)


@dataclass
class BacktestResult:
    """Full backtest results."""
    initial_capital: float
    final_capital: float
    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    n_trades: int
    trades: List[Trade]
    equity_curve: np.ndarray
    signals: np.ndarray
    energies: np.ndarray


# =============================================================================
# Trading Strategy
# =============================================================================


class HNNTradingStrategy:
    """
    Trading strategy based on Hamiltonian Neural Network predictions.

    The strategy:
    1. Observes current phase-space state (q, p)
    2. Integrates forward using symplectic integrator
    3. Predicts future price deviation
    4. Generates signals based on predicted change
    5. Uses energy monitoring for risk management
    """

    def __init__(
        self,
        model: torch.nn.Module,
        prediction_horizon: int = 10,
        integration_dt: float = 0.1,
        entry_threshold: float = 0.5,
        exit_threshold: float = 0.1,
        energy_zscore_limit: float = 2.5,
        max_position_pct: float = 1.0,
        stop_loss_pct: float = 0.03,
        take_profit_pct: float = 0.05,
    ):
        """
        Args:
            model: Trained HNN model.
            prediction_horizon: Number of steps to predict forward.
            integration_dt: Time step for symplectic integration.
            entry_threshold: Minimum |predicted_change| to open position.
            exit_threshold: |predicted_change| below which to close position.
            energy_zscore_limit: Max energy z-score before reducing position.
            max_position_pct: Maximum position as fraction of capital.
            stop_loss_pct: Stop loss percentage.
            take_profit_pct: Take profit percentage.
        """
        self.model = model
        self.prediction_horizon = prediction_horizon
        self.integration_dt = integration_dt
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.energy_zscore_limit = energy_zscore_limit
        self.max_position_pct = max_position_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        self.energy_history: List[float] = []

    def predict_trajectory(
        self,
        q: np.ndarray,
        p: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Predict future trajectory from current state.

        Args:
            q: Current coordinates (1, coord_dim)
            p: Current momenta (1, coord_dim)

        Returns:
            traj_q: Predicted q trajectory
            traj_p: Predicted p trajectory
            current_energy: Current Hamiltonian value
        """
        q_t = torch.FloatTensor(q).unsqueeze(0) if q.ndim == 1 else torch.FloatTensor(q)
        p_t = torch.FloatTensor(p).unsqueeze(0) if p.ndim == 1 else torch.FloatTensor(p)

        # Compute current energy
        with torch.no_grad():
            H = self.model.hamiltonian(q_t, p_t).item()

        # Integrate forward
        with torch.enable_grad():
            traj_q, traj_p = integrate_trajectory(
                self.model,
                q_t,
                p_t,
                dt=self.integration_dt,
                n_steps=self.prediction_horizon,
                method="leapfrog",
            )

        return traj_q.numpy()[:, 0], traj_p.numpy()[:, 0], H

    def compute_energy_zscore(self, energy: float) -> float:
        """Compute z-score of current energy relative to history."""
        self.energy_history.append(energy)

        if len(self.energy_history) < 20:
            return 0.0

        recent = self.energy_history[-100:]  # Use last 100 observations
        mean_e = np.mean(recent)
        std_e = np.std(recent) + 1e-10
        return (energy - mean_e) / std_e

    def generate_signal(
        self,
        q: np.ndarray,
        p: np.ndarray,
    ) -> Tuple[str, float, float, float]:
        """
        Generate trading signal from current phase-space state.

        Returns:
            signal: "BUY", "SELL", or "HOLD"
            strength: Signal strength (|predicted_change|)
            energy: Current energy
            energy_zscore: Energy z-score
        """
        traj_q, traj_p, energy = self.predict_trajectory(q, p)
        energy_zscore = self.compute_energy_zscore(energy)

        # Predicted change in q (price deviation)
        predicted_change = float(traj_q[-1, 0] - traj_q[0, 0])
        predicted_momentum = float(traj_p[-1, 0])
        strength = abs(predicted_change)

        # Check for regime change (energy anomaly)
        if abs(energy_zscore) > self.energy_zscore_limit:
            return "HOLD", strength, energy, energy_zscore

        # Generate signal
        if strength > self.entry_threshold:
            if predicted_change > 0 and predicted_momentum > 0:
                return "BUY", strength, energy, energy_zscore
            elif predicted_change < 0 and predicted_momentum < 0:
                return "SELL", strength, energy, energy_zscore

        if strength < self.exit_threshold:
            return "CLOSE", strength, energy, energy_zscore

        return "HOLD", strength, energy, energy_zscore


# =============================================================================
# Backtester
# =============================================================================


def run_backtest(
    strategy: HNNTradingStrategy,
    prices: np.ndarray,
    q_data: np.ndarray,
    p_data: np.ndarray,
    initial_capital: float = 100000.0,
    commission_pct: float = 0.001,
) -> BacktestResult:
    """
    Run a full backtest of the HNN trading strategy.

    Args:
        strategy: HNNTradingStrategy instance.
        prices: Array of close prices.
        q_data: Phase space q values (normalized).
        p_data: Phase space p values (normalized).
        initial_capital: Starting capital.
        commission_pct: Commission per trade.

    Returns:
        BacktestResult with full performance metrics.
    """
    n = min(len(prices), len(q_data))

    capital = initial_capital
    position = 0.0  # Positive = long, negative = short
    entry_price = 0.0
    trades: List[Trade] = []
    equity_curve = np.zeros(n)
    signals_arr = np.zeros(n)
    energies_arr = np.zeros(n)

    for i in range(n):
        current_price = prices[i]
        q_i = q_data[i].reshape(1, -1)
        p_i = p_data[i].reshape(1, -1)

        # Generate signal
        signal, strength, energy, energy_zscore = strategy.generate_signal(q_i, p_i)
        energies_arr[i] = energy

        # Map signals to numeric
        signal_map = {"BUY": 1, "SELL": -1, "HOLD": 0, "CLOSE": 0}
        signals_arr[i] = signal_map.get(signal, 0)

        # Check stop-loss / take-profit
        if position != 0:
            pnl_pct = (current_price - entry_price) / entry_price
            if position < 0:
                pnl_pct = -pnl_pct

            if pnl_pct <= -strategy.stop_loss_pct:
                signal = "CLOSE"
            elif pnl_pct >= strategy.take_profit_pct:
                signal = "CLOSE"

        # Execute signal
        if signal == "BUY" and position <= 0:
            if position < 0:
                # Close short
                pnl = position * (current_price - entry_price)
                capital += pnl - abs(position) * current_price * commission_pct
                trades.append(Trade(
                    timestamp=i,
                    side="CLOSE_SHORT",
                    price=current_price,
                    quantity=abs(position),
                    signal_strength=strength,
                    energy=energy,
                    energy_zscore=energy_zscore,
                ))

            # Open long
            qty = (capital * strategy.max_position_pct) / current_price
            position = qty
            entry_price = current_price
            capital -= qty * current_price * commission_pct
            trades.append(Trade(
                timestamp=i,
                side="BUY",
                price=current_price,
                quantity=qty,
                signal_strength=strength,
                energy=energy,
                energy_zscore=energy_zscore,
            ))

        elif signal == "SELL" and position >= 0:
            if position > 0:
                # Close long
                pnl = position * (current_price - entry_price)
                capital += pnl - position * current_price * commission_pct
                trades.append(Trade(
                    timestamp=i,
                    side="CLOSE_LONG",
                    price=current_price,
                    quantity=position,
                    signal_strength=strength,
                    energy=energy,
                    energy_zscore=energy_zscore,
                ))

            # Open short
            qty = (capital * strategy.max_position_pct) / current_price
            position = -qty
            entry_price = current_price
            capital -= qty * current_price * commission_pct
            trades.append(Trade(
                timestamp=i,
                side="SELL",
                price=current_price,
                quantity=qty,
                signal_strength=strength,
                energy=energy,
                energy_zscore=energy_zscore,
            ))

        elif signal == "CLOSE" and position != 0:
            pnl = position * (current_price - entry_price)
            if position < 0:
                pnl = -pnl
            capital += pnl - abs(position) * current_price * commission_pct

            side = "CLOSE_LONG" if position > 0 else "CLOSE_SHORT"
            trades.append(Trade(
                timestamp=i,
                side=side,
                price=current_price,
                quantity=abs(position),
                signal_strength=strength,
                energy=energy,
                energy_zscore=energy_zscore,
            ))
            position = 0.0

        # Update equity curve
        if position > 0:
            equity_curve[i] = capital + position * (current_price - entry_price)
        elif position < 0:
            equity_curve[i] = capital - position * (current_price - entry_price)
        else:
            equity_curve[i] = capital

    # Close any remaining position
    if position != 0:
        final_price = prices[n - 1]
        pnl = position * (final_price - entry_price)
        if position < 0:
            pnl = -pnl
        capital += pnl

    # Compute metrics
    total_return = (capital - initial_capital) / initial_capital

    # Annualized return (assume 252 trading days)
    n_days = n / 288 if n > 288 else max(n, 1)  # ~288 5-min candles per day
    annualized_return = (1 + total_return) ** (365 / max(n_days, 1)) - 1

    # Maximum drawdown
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / (peak + 1e-10)
    max_drawdown = float(drawdown.min())

    # Sharpe ratio
    returns = np.diff(equity_curve) / (equity_curve[:-1] + 1e-10)
    sharpe = (
        np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252 * 288)
        if len(returns) > 0
        else 0.0
    )

    # Win rate
    pnl_per_trade = []
    for i in range(0, len(trades) - 1, 2):
        if i + 1 < len(trades):
            if trades[i].side in ("BUY", "SELL"):
                entry_p = trades[i].price
                exit_p = trades[i + 1].price
                if trades[i].side == "BUY":
                    pnl_per_trade.append(exit_p - entry_p)
                else:
                    pnl_per_trade.append(entry_p - exit_p)

    win_rate = (
        sum(1 for p in pnl_per_trade if p > 0) / max(len(pnl_per_trade), 1)
    )

    return BacktestResult(
        initial_capital=initial_capital,
        final_capital=capital,
        total_return=total_return,
        annualized_return=annualized_return,
        max_drawdown=max_drawdown,
        sharpe_ratio=float(sharpe),
        win_rate=win_rate,
        n_trades=len(trades),
        trades=trades,
        equity_curve=equity_curve,
        signals=signals_arr,
        energies=energies_arr,
    )


def plot_backtest_results(
    result: BacktestResult,
    prices: np.ndarray,
    save_path: str = None,
):
    """Plot comprehensive backtest results."""
    fig, axes = plt.subplots(4, 1, figsize=(16, 16), sharex=True)

    n = len(result.equity_curve)

    # 1. Price and signals
    ax1 = axes[0]
    ax1.plot(prices[:n], "k-", linewidth=0.8, label="Price")
    buy_idx = np.where(result.signals == 1)[0]
    sell_idx = np.where(result.signals == -1)[0]
    ax1.scatter(
        buy_idx,
        prices[buy_idx],
        c="green",
        marker="^",
        s=50,
        alpha=0.7,
        label="Buy",
        zorder=5,
    )
    ax1.scatter(
        sell_idx,
        prices[sell_idx],
        c="red",
        marker="v",
        s=50,
        alpha=0.7,
        label="Sell",
        zorder=5,
    )
    ax1.set_ylabel("Price")
    ax1.set_title("Price and Trading Signals")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Equity curve
    ax2 = axes[1]
    ax2.plot(result.equity_curve, "b-", linewidth=1)
    ax2.axhline(
        y=result.initial_capital, color="gray", linestyle="--", alpha=0.5
    )
    ax2.fill_between(
        range(n),
        result.initial_capital,
        result.equity_curve,
        alpha=0.1,
        color="blue",
    )
    ax2.set_ylabel("Capital")
    ax2.set_title(
        f"Equity Curve (Return: {result.total_return*100:.2f}%, "
        f"Sharpe: {result.sharpe_ratio:.2f})"
    )
    ax2.grid(True, alpha=0.3)

    # 3. Drawdown
    ax3 = axes[2]
    peak = np.maximum.accumulate(result.equity_curve)
    drawdown = (result.equity_curve - peak) / (peak + 1e-10) * 100
    ax3.fill_between(range(n), 0, drawdown, color="red", alpha=0.3)
    ax3.plot(drawdown, "r-", linewidth=0.5)
    ax3.set_ylabel("Drawdown (%)")
    ax3.set_title(f"Drawdown (Max: {result.max_drawdown*100:.2f}%)")
    ax3.grid(True, alpha=0.3)

    # 4. Energy monitoring
    ax4 = axes[3]
    ax4.plot(result.energies, "purple", linewidth=0.5, alpha=0.7)
    ax4.set_ylabel("H(q, p)")
    ax4.set_xlabel("Time step")
    ax4.set_title("Hamiltonian Energy (Regime Monitor)")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Backtest HNN trading strategy")
    parser.add_argument("--model", type=str, required=True, help="Path to saved model")
    parser.add_argument("--source", type=str, default="bybit")
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--interval", type=str, default="5")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--ma-window", type=int, default=20)
    parser.add_argument("--initial-capital", type=float, default=100000.0)
    parser.add_argument("--commission", type=float, default=0.001)
    parser.add_argument("--prediction-horizon", type=int, default=10)
    parser.add_argument("--entry-threshold", type=float, default=0.5)
    parser.add_argument("--stop-loss", type=float, default=0.03)
    parser.add_argument("--take-profit", type=float, default=0.05)
    parser.add_argument("--output-dir", type=str, default="output")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

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
    print(f"Loaded {model_type} model (val_loss={checkpoint['val_loss']:.6f})")

    # Fetch data
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

    # Construct phase space
    q, p, dq_dt, dp_dt = construct_phase_space(
        df, ma_window=args.ma_window, momentum_type="returns"
    )
    q_norm, p_norm, _, _, stats = normalize_phase_space(q, p, dq_dt, dp_dt)

    # Get prices aligned with phase space
    valid_start = args.ma_window
    prices = df["close"].values[valid_start : valid_start + len(q_norm)]

    # Create strategy
    strategy = HNNTradingStrategy(
        model=model,
        prediction_horizon=args.prediction_horizon,
        integration_dt=0.1,
        entry_threshold=args.entry_threshold,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
    )

    # Run backtest
    print("\nRunning backtest...")
    result = run_backtest(
        strategy=strategy,
        prices=prices,
        q_data=q_norm,
        p_data=p_norm,
        initial_capital=args.initial_capital,
        commission_pct=args.commission,
    )

    # Print results
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"Initial Capital:     ${result.initial_capital:,.2f}")
    print(f"Final Capital:       ${result.final_capital:,.2f}")
    print(f"Total Return:        {result.total_return*100:+.2f}%")
    print(f"Annualized Return:   {result.annualized_return*100:+.2f}%")
    print(f"Max Drawdown:        {result.max_drawdown*100:.2f}%")
    print(f"Sharpe Ratio:        {result.sharpe_ratio:.3f}")
    print(f"Win Rate:            {result.win_rate*100:.1f}%")
    print(f"Number of Trades:    {result.n_trades}")
    print("=" * 60)

    # Plot results
    plot_path = os.path.join(args.output_dir, "backtest_results.png")
    plot_backtest_results(result, prices, save_path=plot_path)

    # Save results to JSON
    results_dict = {
        "initial_capital": result.initial_capital,
        "final_capital": result.final_capital,
        "total_return": result.total_return,
        "annualized_return": result.annualized_return,
        "max_drawdown": result.max_drawdown,
        "sharpe_ratio": result.sharpe_ratio,
        "win_rate": result.win_rate,
        "n_trades": result.n_trades,
        "strategy_params": {
            "prediction_horizon": args.prediction_horizon,
            "entry_threshold": args.entry_threshold,
            "stop_loss": args.stop_loss,
            "take_profit": args.take_profit,
        },
    }
    results_path = os.path.join(args.output_dir, "backtest_results.json")
    with open(results_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
