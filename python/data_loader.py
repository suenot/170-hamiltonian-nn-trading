"""
Data loading and phase space construction for HNN trading.

Supports:
  - Bybit V5 API (cryptocurrency: BTC/USDT, ETH/USDT, etc.)
  - Yahoo Finance (stocks: SPY, AAPL, etc.)

Constructs phase space (q, p) from OHLCV data:
  q: generalized coordinates (price deviation from moving average)
  p: conjugate momenta (returns, volume-weighted momentum)
"""

import argparse
import os
import time
import numpy as np
import pandas as pd
import requests
from typing import Tuple, Optional, Dict, List


# =============================================================================
# Bybit Data Fetching
# =============================================================================


def fetch_bybit_klines(
    symbol: str = "BTCUSDT",
    interval: str = "5",
    limit: int = 1000,
    category: str = "linear",
    end_time: Optional[int] = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV kline data from Bybit V5 API.

    Args:
        symbol: Trading pair symbol (e.g., "BTCUSDT").
        interval: Kline interval in minutes ("1", "3", "5", "15", "30", "60", "240", "D", "W").
        limit: Number of candles to fetch (max 1000 per request).
        category: Market category ("linear" for perpetual futures, "spot" for spot).
        end_time: End timestamp in milliseconds (optional).

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume, turnover
    """
    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "category": category,
        "symbol": symbol,
        "interval": interval,
        "limit": min(limit, 1000),
    }
    if end_time is not None:
        params["end"] = str(end_time)

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    if data["retCode"] != 0:
        raise ValueError(f"Bybit API error: {data['retMsg']}")

    rows = data["result"]["list"]
    df = pd.DataFrame(
        rows,
        columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"],
    )

    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        df[col] = df[col].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def fetch_bybit_extended(
    symbol: str = "BTCUSDT",
    interval: str = "5",
    total_candles: int = 5000,
    category: str = "linear",
) -> pd.DataFrame:
    """
    Fetch extended history from Bybit by paginating through the API.

    Args:
        symbol: Trading pair.
        interval: Candle interval.
        total_candles: Total number of candles to fetch.
        category: Market category.

    Returns:
        DataFrame with full OHLCV history.
    """
    all_dfs: List[pd.DataFrame] = []
    end_time = None
    remaining = total_candles

    while remaining > 0:
        batch_size = min(remaining, 1000)
        df = fetch_bybit_klines(
            symbol=symbol,
            interval=interval,
            limit=batch_size,
            category=category,
            end_time=end_time,
        )

        if len(df) == 0:
            break

        all_dfs.append(df)
        remaining -= len(df)

        # Set end_time to earliest timestamp for next batch
        earliest_ts = df["timestamp"].min()
        end_time = int(earliest_ts.timestamp() * 1000) - 1

        # Rate limiting
        time.sleep(0.1)

    if not all_dfs:
        return pd.DataFrame()

    result = pd.concat(all_dfs, ignore_index=True)
    result = result.drop_duplicates(subset=["timestamp"])
    result = result.sort_values("timestamp").reset_index(drop=True)
    return result


# =============================================================================
# Yahoo Finance Data Fetching
# =============================================================================


def fetch_yahoo_data(
    symbol: str = "SPY",
    period: str = "2y",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Fetch stock data from Yahoo Finance.

    Args:
        symbol: Stock ticker (e.g., "SPY", "AAPL").
        period: Data period ("1mo", "3mo", "6mo", "1y", "2y", "5y").
        interval: Data interval ("1d", "1h", "5m").

    Returns:
        DataFrame with OHLCV data.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is required: pip install yfinance")

    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)
    df = df.reset_index()

    # Normalize column names
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    if "date" in df.columns:
        df = df.rename(columns={"date": "timestamp"})
    elif "datetime" in df.columns:
        df = df.rename(columns={"datetime": "timestamp"})

    # Keep only needed columns
    keep_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    df = df[[c for c in keep_cols if c in df.columns]]

    return df


# =============================================================================
# Phase Space Construction
# =============================================================================


def construct_phase_space(
    df: pd.DataFrame,
    ma_window: int = 20,
    momentum_type: str = "returns",
    use_volume: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert OHLCV price data to (q, p, dq/dt, dp/dt) phase space.

    Args:
        df: DataFrame with at least 'close' column.
        ma_window: Window for moving average (defines equilibrium).
        momentum_type: How to compute momentum:
            - "returns": Simple log returns
            - "volume_weighted": Returns weighted by volume
            - "rsi_deviation": RSI - 50
        use_volume: If True, add volume deviation as additional q dimension.

    Returns:
        q: Generalized coordinates, shape (N, q_dim)
        p: Conjugate momenta, shape (N, p_dim)
        dq_dt: Time derivative of q, shape (N, q_dim)
        dp_dt: Time derivative of p, shape (N, p_dim)
    """
    close = df["close"].values
    log_close = np.log(close)

    # q1: Price deviation from moving average
    ma = pd.Series(log_close).rolling(ma_window).mean().values
    q_price = log_close - ma

    # Build q and p arrays
    q_cols: List[np.ndarray] = [q_price]
    p_cols: List[np.ndarray] = []

    if use_volume and "volume" in df.columns:
        log_vol = np.log(df["volume"].values + 1)
        vol_ma = pd.Series(log_vol).rolling(ma_window).mean().values
        q_vol = log_vol - vol_ma
        q_cols.append(q_vol)

    # p: momentum
    if momentum_type == "returns":
        p_price = np.gradient(log_close)
        p_cols.append(p_price)
    elif momentum_type == "volume_weighted":
        returns = np.gradient(log_close)
        if "volume" in df.columns:
            vol_norm = df["volume"].values / (
                pd.Series(df["volume"].values).rolling(ma_window).mean().values + 1e-10
            )
            p_price = returns * vol_norm
        else:
            p_price = returns
        p_cols.append(p_price)
    elif momentum_type == "rsi_deviation":
        rsi = compute_rsi(close, period=14)
        p_price = (rsi - 50.0) / 50.0  # Normalize to [-1, 1]
        p_cols.append(p_price)
    else:
        raise ValueError(f"Unknown momentum_type: {momentum_type}")

    if use_volume and "volume" in df.columns:
        vol_mom = np.gradient(np.log(df["volume"].values + 1))
        p_cols.append(vol_mom)

    q = np.column_stack(q_cols)
    p = np.column_stack(p_cols)

    # Compute time derivatives
    dq_dt = np.gradient(q, axis=0)
    dp_dt = np.gradient(p, axis=0)

    # Remove NaN rows (from moving average warmup)
    valid = np.all(np.isfinite(q), axis=1) & np.all(np.isfinite(p), axis=1)
    valid &= np.all(np.isfinite(dq_dt), axis=1) & np.all(np.isfinite(dp_dt), axis=1)

    return q[valid], p[valid], dq_dt[valid], dp_dt[valid]


def construct_multiscale_phase_space(
    df: pd.DataFrame,
    windows: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct multi-scale phase space with multiple MA windows.

    Args:
        df: DataFrame with 'close' column.
        windows: List of MA windows (e.g., [5, 20, 50]).

    Returns:
        q, p, dq_dt, dp_dt arrays with dimension = len(windows).
    """
    if windows is None:
        windows = [5, 20, 50]

    close = df["close"].values
    log_close = np.log(close)

    q_cols = []
    p_cols = []

    for w in windows:
        ma = pd.Series(log_close).rolling(w).mean().values
        q_dev = log_close - ma
        q_cols.append(q_dev)

        # Momentum at this scale
        returns_w = pd.Series(log_close).diff(w).values / w
        p_cols.append(returns_w)

    q = np.column_stack(q_cols)
    p = np.column_stack(p_cols)

    dq_dt = np.gradient(q, axis=0)
    dp_dt = np.gradient(p, axis=0)

    valid = np.all(np.isfinite(q), axis=1) & np.all(np.isfinite(p), axis=1)
    valid &= np.all(np.isfinite(dq_dt), axis=1) & np.all(np.isfinite(dp_dt), axis=1)

    return q[valid], p[valid], dq_dt[valid], dp_dt[valid]


# =============================================================================
# Feature Engineering
# =============================================================================


def compute_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Compute Relative Strength Index."""
    deltas = np.diff(prices, prepend=prices[0])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = pd.Series(gains).rolling(period).mean().values
    avg_loss = pd.Series(losses).rolling(period).mean().values

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi


def normalize_phase_space(
    q: np.ndarray,
    p: np.ndarray,
    dq_dt: np.ndarray,
    dp_dt: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Normalize phase space to zero mean, unit variance.

    Returns:
        Normalized (q, p, dq_dt, dp_dt) and stats dict for denormalization.
    """
    stats = {
        "q_mean": q.mean(0),
        "q_std": q.std(0) + 1e-8,
        "p_mean": p.mean(0),
        "p_std": p.std(0) + 1e-8,
    }

    q_norm = (q - stats["q_mean"]) / stats["q_std"]
    p_norm = (p - stats["p_mean"]) / stats["p_std"]
    dq_dt_norm = dq_dt / stats["q_std"]
    dp_dt_norm = dp_dt / stats["p_std"]

    return q_norm, p_norm, dq_dt_norm, dp_dt_norm, stats


def denormalize_phase_space(
    q_norm: np.ndarray,
    p_norm: np.ndarray,
    stats: Dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reverse normalization."""
    q = q_norm * stats["q_std"] + stats["q_mean"]
    p = p_norm * stats["p_std"] + stats["p_mean"]
    return q, p


def train_test_split_sequential(
    q: np.ndarray,
    p: np.ndarray,
    dq_dt: np.ndarray,
    dp_dt: np.ndarray,
    train_ratio: float = 0.8,
) -> Tuple:
    """
    Split phase space data into train/test (sequential, no shuffle).

    Returns:
        (q_train, p_train, dq_train, dp_train,
         q_test, p_test, dq_test, dp_test)
    """
    n = len(q)
    split = int(n * train_ratio)
    return (
        q[:split],
        p[:split],
        dq_dt[:split],
        dp_dt[:split],
        q[split:],
        p[split:],
        dq_dt[split:],
        dp_dt[split:],
    )


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Fetch market data for HNN trading")
    parser.add_argument(
        "--source",
        type=str,
        default="bybit",
        choices=["bybit", "yahoo"],
        help="Data source",
    )
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading symbol")
    parser.add_argument(
        "--interval", type=str, default="5", help="Candle interval (minutes or D/W)"
    )
    parser.add_argument(
        "--limit", type=int, default=5000, help="Number of candles to fetch"
    )
    parser.add_argument(
        "--ma-window", type=int, default=20, help="Moving average window"
    )
    parser.add_argument(
        "--output", type=str, default="data", help="Output directory"
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Fetching data from {args.source}: {args.symbol}...")

    if args.source == "bybit":
        df = fetch_bybit_extended(
            symbol=args.symbol,
            interval=args.interval,
            total_candles=args.limit,
        )
    else:
        period_map = {
            5000: "5y",
            2000: "2y",
            1000: "1y",
            500: "6mo",
        }
        period = "2y"
        for threshold, p in sorted(period_map.items()):
            if args.limit <= threshold:
                period = p
                break
        df = fetch_yahoo_data(symbol=args.symbol, period=period, interval="1d")

    print(f"Fetched {len(df)} candles")
    print(f"Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
    print(f"Price range: {df['close'].min():.2f} to {df['close'].max():.2f}")

    # Save raw data
    raw_path = os.path.join(args.output, f"{args.symbol}_raw.csv")
    df.to_csv(raw_path, index=False)
    print(f"Saved raw data to {raw_path}")

    # Construct phase space
    print("\nConstructing phase space...")
    q, p, dq_dt, dp_dt = construct_phase_space(
        df, ma_window=args.ma_window, momentum_type="returns"
    )
    print(f"Phase space shape: q={q.shape}, p={p.shape}")
    print(f"q range: [{q.min():.6f}, {q.max():.6f}]")
    print(f"p range: [{p.min():.6f}, {p.max():.6f}]")

    # Normalize
    q_norm, p_norm, dq_norm, dp_norm, stats = normalize_phase_space(
        q, p, dq_dt, dp_dt
    )

    # Save phase space data
    phase_path = os.path.join(args.output, f"{args.symbol}_phase_space.npz")
    np.savez(
        phase_path,
        q=q_norm,
        p=p_norm,
        dq_dt=dq_norm,
        dp_dt=dp_norm,
        q_raw=q,
        p_raw=p,
        **{f"stats_{k}": v for k, v in stats.items()},
    )
    print(f"Saved phase space to {phase_path}")

    # Train/test split
    splits = train_test_split_sequential(q_norm, p_norm, dq_norm, dp_norm)
    print(f"\nTrain samples: {len(splits[0])}")
    print(f"Test samples:  {len(splits[4])}")


if __name__ == "__main__":
    main()
