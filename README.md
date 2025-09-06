# Chapter 149: Hamiltonian Neural Networks for Trading

## Overview

Hamiltonian Neural Networks (HNNs) bring physics-informed inductive biases into financial modeling by learning conserved quantities from market data. Instead of treating price dynamics as arbitrary time series, HNNs model them as Hamiltonian systems where "energy" is conserved -- momentum builds up, converts to price changes, and cycles back. This chapter shows how to build, train, and deploy HNNs for trading on both stock and cryptocurrency markets (Bybit exchange).

**Key Insight:** Markets exhibit quasi-Hamiltonian dynamics -- momentum and mean reversion are two sides of the same energy conservation coin. By learning the Hamiltonian function H(q, p) from data, we obtain dynamics that are inherently stable over long horizons and respect the natural phase-space structure of price-momentum interactions.

## Trading Strategy

**Core Strategy:** Learn the Hamiltonian of a price-momentum phase space, then use symplectic integration to predict future trajectories. Trade when predicted trajectories diverge from current prices by more than a threshold.

**Edge Factors:**
1. Energy conservation prevents unbounded prediction drift (a common failure mode of standard RNNs/MLPs)
2. Symplectic structure preserves phase-space volume, avoiding artificial trajectory collapse
3. Dissipative extensions capture transaction costs and market friction realistically
4. Long-horizon stability enables multi-step lookahead strategies

**Target Assets:** Cryptocurrency pairs (BTC/USDT, ETH/USDT) from Bybit exchange, plus traditional equities via Yahoo Finance.

---

## Hamiltonian Mechanics Primer

### Classical Hamiltonian Mechanics

Hamiltonian mechanics describes the evolution of a physical system using generalized coordinates q (positions) and conjugate momenta p. The Hamiltonian function H(q, p) represents the total energy of the system:

```
H(q, p) = T(p) + V(q)

where:
  T(p) = kinetic energy (function of momenta)
  V(q) = potential energy (function of positions)
```

The system evolves according to Hamilton's equations:

```
dq/dt =  dH/dp    (velocity = derivative of H w.r.t. momentum)
dp/dt = -dH/dq    (force = negative derivative of H w.r.t. position)
```

### Why This Matters

These equations have a fundamental property: **they conserve the Hamiltonian**. If you compute dH/dt along a trajectory:

```
dH/dt = (dH/dq)(dq/dt) + (dH/dp)(dp/dt)
      = (dH/dq)(dH/dp) + (dH/dp)(-dH/dq)
      = 0
```

The total energy is exactly conserved. This is not a numerical trick -- it is a structural property of the equations themselves.

### Phase Space and Symplectic Structure

The state space (q, p) is called **phase space**. Hamiltonian flows preserve the symplectic structure -- intuitively, they preserve the "area" in phase space (Liouville's theorem). This means:

```
Phase Space Evolution:
                    p (momentum)
                    ^
                    |    .-'''-.
                    |   /       \
                    |  |    *--->|   Trajectories form
                    |   \       /   closed orbits (conservative)
                    |    '-...-'    or spirals (dissipative)
                    |
                    +--------------> q (position)
```

For a simple harmonic oscillator (spring), trajectories are ellipses in phase space. The system oscillates forever between kinetic and potential energy.

### The Pendulum Analogy for Markets

Consider a pendulum:
- **Position q**: angle of displacement (analogous to price deviation from mean)
- **Momentum p**: angular velocity (analogous to price momentum/rate of change)
- **Potential energy V(q)**: gravitational energy (analogous to mean-reversion force)
- **Kinetic energy T(p)**: rotational energy (analogous to momentum energy)

```
Pendulum Phase Portrait:          Market Phase Portrait:
        p                                 momentum
        ^                                    ^
        |   .--.                             |   .--.
        |  / .. \                            |  / .. \
        | | .  . |                           | | .  . |
   -----+--'----'----> q              ------+--'----'----> price deviation
        | | .  . |                           | | .  . |
        |  \ .. /                            |  \ .. /
        |   '--'                             |   '--'
```

A mean-reverting asset behaves like a pendulum: price deviates (potential energy increases), momentum builds to push it back (kinetic energy), it overshoots, and the cycle repeats.

---

## Hamiltonian Neural Networks (HNNs)

### The Core Idea

Instead of learning dynamics directly (as in standard Neural ODEs), HNNs learn the **Hamiltonian function** H_theta(q, p) using a neural network. The dynamics are then derived via automatic differentiation:

```
Standard Neural ODE:        Hamiltonian Neural Network:

  dx/dt = f_theta(x)          Learn: H_theta(q, p)
                               Derive: dq/dt =  dH_theta/dp
  (no structure)                       dp/dt = -dH_theta/dq
                               (energy conservation guaranteed)
```

### Architecture

```
                    ┌─────────────────────────────┐
  q (position) ──> │                               │
                    │   Neural Network H_theta      │ ──> H (scalar)
  p (momentum) ──> │   (MLP with smooth activations)│
                    └─────────────────────────────┘
                                 │
                         autograd │
                    ┌────────────┴────────────┐
                    │                          │
                    v                          v
              dH/dp = dq/dt             -dH/dq = dp/dt
```

**Key design choices:**
1. **Smooth activations** (tanh, softplus) -- Hamilton's equations require second derivatives to be well-defined
2. **Scalar output** -- H_theta outputs a single number (the energy)
3. **Autograd** -- derivatives are computed exactly via backpropagation, not finite differences

### Mathematical Formulation

Given a dataset of state observations {(q_i, p_i, dq_i/dt, dp_i/dt)}, we train:

```
Loss = sum_i || dH_theta/dp_i - dq_i/dt ||^2 + || -dH_theta/dq_i - dp_i/dt ||^2
```

This loss says: "the derivatives of our learned Hamiltonian should match the observed dynamics."

### Code: Basic HNN in PyTorch

```python
import torch
import torch.nn as nn

class HamiltonianNN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=3):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, q, p):
        """Compute the Hamiltonian H(q, p)."""
        x = torch.cat([q, p], dim=-1)
        return self.net(x)

    def time_derivative(self, q, p):
        """Compute dq/dt, dp/dt via Hamilton's equations."""
        q = q.requires_grad_(True)
        p = p.requires_grad_(True)
        H = self.forward(q, p)

        dH = torch.autograd.grad(H.sum(), [q, p], create_graph=True)
        dH_dq, dH_dp = dH[0], dH[1]

        dq_dt = dH_dp      # Hamilton's first equation
        dp_dt = -dH_dq     # Hamilton's second equation
        return dq_dt, dp_dt
```

---

## Symplectic Integration

### Why Standard Integrators Fail

Standard numerical integrators (Euler, RK4) do not preserve the symplectic structure. Over long time horizons, they accumulate energy drift:

```
Energy over time:

  Standard RK4:                Symplectic Leapfrog:
  H │\                         H │
    │ \___                       │ ~~~~~~~~~~~~~~~~~~~~
    │      \___                  │
    │          \___              │
    │              \             │
    └──────────────> t           └──────────────────> t
  (energy drifts down)         (energy oscillates, no drift)
```

### Leapfrog / Stormer-Verlet Integrator

The leapfrog integrator is symplectic -- it exactly preserves the phase-space structure:

```
Algorithm: Leapfrog Integration

Given: (q_0, p_0), step size dt, number of steps N

For each step n = 0, 1, ..., N-1:
    p_{n+1/2} = p_n     - (dt/2) * dH/dq(q_n, p_n)         # half-step momentum
    q_{n+1}   = q_n     + dt     * dH/dp(q_{n+1/2}, p_{n+1/2})  # full-step position
    p_{n+1}   = p_{n+1/2} - (dt/2) * dH/dq(q_{n+1}, p_{n+1/2}) # half-step momentum
```

**Properties:**
- Time-reversible
- Symplectic (preserves phase-space volume)
- Energy error is bounded and oscillatory (no secular drift)
- Second-order accurate

### Code: Symplectic Integrator

```python
def leapfrog_step(model, q, p, dt):
    """One step of leapfrog/Stormer-Verlet integration."""
    # Half-step momentum
    dq_dt, dp_dt = model.time_derivative(q, p)
    p_half = p + 0.5 * dt * dp_dt

    # Full-step position
    dq_dt, _ = model.time_derivative(q, p_half)
    q_new = q + dt * dq_dt

    # Half-step momentum
    _, dp_dt = model.time_derivative(q_new, p_half)
    p_new = p_half + 0.5 * dt * dp_dt

    return q_new, p_new

def integrate_trajectory(model, q0, p0, dt, n_steps):
    """Integrate a trajectory using leapfrog."""
    trajectory_q = [q0]
    trajectory_p = [p0]
    q, p = q0, p0
    for _ in range(n_steps):
        q, p = leapfrog_step(model, q, p, dt)
        trajectory_q.append(q)
        trajectory_p.append(p)
    return torch.stack(trajectory_q), torch.stack(trajectory_p)
```

---

## Dissipative Hamiltonian Neural Networks

### Markets Are Not Perfectly Conservative

Real markets have friction: transaction costs, slippage, information decay, and regime changes. A pure Hamiltonian system conserves energy exactly, but markets lose "energy" over time. We need **dissipative** extensions.

### Dissipative HNN Formulation

A dissipative Hamiltonian system adds a Rayleigh dissipation function D(q, p):

```
dq/dt =  dH/dp
dp/dt = -dH/dq - dD/dp

where D(q, p) >= 0 is the dissipation function (always non-negative)
```

The total energy now decreases over time:

```
dH/dt = -p^T * dD/dp <= 0   (energy can only decrease)
```

### Architecture

```
                ┌──────────────────┐
  (q, p) ────> │  H_theta (MLP)   │ ──> H (Hamiltonian, scalar)
                └──────────────────┘
                         │ autograd
                         v
                ┌──────────────────┐
  (q, p) ────> │  D_phi (MLP)     │ ──> D (Dissipation, scalar >= 0)
                └──────────────────┘
                         │ autograd
                         v
              dq/dt = dH/dp
              dp/dt = -dH/dq - dD/dp
```

### Code: Dissipative HNN

```python
class DissipativeHNN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64):
        super().__init__()
        # Hamiltonian network
        self.H_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        # Dissipation network (output must be non-negative)
        self.D_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # ensures D >= 0
        )

    def hamiltonian(self, q, p):
        x = torch.cat([q, p], dim=-1)
        return self.H_net(x)

    def dissipation(self, q, p):
        x = torch.cat([q, p], dim=-1)
        return self.D_net(x)

    def time_derivative(self, q, p):
        q = q.requires_grad_(True)
        p = p.requires_grad_(True)

        H = self.hamiltonian(q, p)
        dH = torch.autograd.grad(H.sum(), [q, p], create_graph=True)
        dH_dq, dH_dp = dH

        D = self.dissipation(q, p)
        dD_dp = torch.autograd.grad(D.sum(), p, create_graph=True)[0]

        dq_dt = dH_dp
        dp_dt = -dH_dq - dD_dp  # dissipation subtracts from momentum
        return dq_dt, dp_dt
```

---

## Port-Hamiltonian Neural Networks for Open Market Systems

### Markets as Open Systems

Markets are not isolated -- they receive external inputs (news, monetary policy, capital flows). Port-Hamiltonian systems model this:

```
dq/dt =  dH/dp + g_q(q, p) * u(t)
dp/dt = -dH/dq + g_p(q, p) * u(t) - dD/dp

where:
  u(t) = external input (news sentiment, volume shocks, etc.)
  g(q, p) = input coupling matrix
```

This framework cleanly separates:
1. **Internal dynamics** (Hamiltonian + dissipation)
2. **External forcing** (port connections)
3. **Energy accounting** (energy in from ports, energy dissipated, energy stored)

### Application to Market Microstructure

```
Port-Hamiltonian Market Model:

  External Inputs u(t):                Internal Dynamics:
  ┌─────────────────┐                  ┌─────────────────────────┐
  │ Order flow       │──────────────>  │                          │
  │ News sentiment   │──────────────>  │  H(q, p) + D(q, p)     │
  │ Volume shocks    │──────────────>  │                          │
  │ Macro indicators │──────────────>  │  q = price deviation     │
  └─────────────────┘                  │  p = order book momentum │
                                       └──────────┬──────────────┘
                                                   │
                                                   v
                                         Price trajectory (q(t), p(t))
```

---

## Application to Financial Markets

### Phase Space Construction for Trading

To apply HNNs to trading, we need to define appropriate generalized coordinates:

```
Generalized coordinates q (position-like):
  - Log price deviation from moving average: q = log(P) - MA(log(P))
  - Normalized price: q = (P - mean) / std
  - Spread in pairs trading: q = log(P1/P2) - mean

Conjugate momenta p (momentum-like):
  - Rate of price change: p = d(log P)/dt (returns)
  - Volume-weighted momentum: p = return * volume
  - Order flow imbalance: p = (buy_volume - sell_volume) / total_volume
  - RSI deviation: p = RSI - 50
```

### Why Markets Are Quasi-Hamiltonian

Several market phenomena map naturally to Hamiltonian dynamics:

**1. Mean Reversion as Harmonic Oscillator**
```
H(q, p) = p^2/2 + k*q^2/2

dq/dt = p              (momentum drives price)
dp/dt = -k*q           (deviation creates restoring force)

Period = 2*pi/sqrt(k)  (mean reversion timescale)
```

**2. Momentum-Reversal Cycles**
```
Phase 1: Momentum builds (p increases, q moves away from equilibrium)
Phase 2: Reversal force grows (dH/dq increases)
Phase 3: Momentum decays (p decreases, reversal begins)
Phase 4: Price reverts (q returns toward equilibrium)

This is exactly a Hamiltonian orbit in (q, p) space.
```

**3. Market Microstructure**
```
Order book has natural Hamiltonian structure:
  - q = mid-price deviation
  - p = order flow imbalance
  - V(q) = liquidity provider pressure (mean-reverting)
  - T(p) = momentum trader pressure (trend-following)
  - D(p) = bid-ask spread friction (dissipation)
```

### Constructing Training Data

```python
def construct_phase_space(prices, window=20):
    """Convert price series to (q, p) phase space."""
    log_prices = np.log(prices)

    # q: deviation from moving average
    ma = pd.Series(log_prices).rolling(window).mean().values
    q = log_prices - ma

    # p: rate of change (momentum)
    p = np.gradient(log_prices)

    # Time derivatives (for training targets)
    dq_dt = np.gradient(q)
    dp_dt = np.gradient(p)

    # Remove NaN edges
    valid = ~(np.isnan(q) | np.isnan(p) | np.isnan(dq_dt) | np.isnan(dp_dt))

    return q[valid], p[valid], dq_dt[valid], dp_dt[valid]
```

---

## Comparison with Standard Neural ODEs

| Feature | Standard Neural ODE | Hamiltonian NN |
|---------|-------------------|----------------|
| Learned function | f_theta(x) (vector field) | H_theta(q,p) (scalar energy) |
| Conservation laws | None guaranteed | Energy conserved by construction |
| Long-horizon stability | Drift accumulates | Bounded, oscillatory errors |
| Parameter efficiency | O(d^2) for d-dim | O(d^2) but scalar output constrains |
| Phase space structure | None | Symplectic (area-preserving) |
| Physical interpretability | Low | High (energy decomposition) |
| Integration | Any ODE solver | Symplectic integrators preferred |
| Dissipation | Must be learned implicitly | Explicit dissipation term |
| External inputs | Ad-hoc | Port-Hamiltonian framework |

### When to Use HNNs vs Neural ODEs

**Use HNNs when:**
- The system exhibits oscillatory/mean-reverting behavior
- Long-horizon predictions are needed
- Physical interpretability matters (energy analysis)
- Training data is limited (stronger inductive bias helps)

**Use standard Neural ODEs when:**
- Dynamics are highly non-conservative (trending markets)
- The system has no clear phase-space decomposition
- Very high-dimensional state spaces
- Maximum flexibility is needed

---

## Bybit Cryptocurrency Application

### Market Microstructure as Hamiltonian System

Cryptocurrency markets on Bybit have features that make them well-suited for Hamiltonian modeling:

1. **24/7 trading** -- continuous dynamics, no overnight gaps
2. **High frequency data** -- rich phase-space trajectories
3. **Mean reversion at short timescales** -- market-making around fair value
4. **Momentum at medium timescales** -- trend-following behavior
5. **Perpetual futures funding rate** -- explicit mean-reversion mechanism

### Data Pipeline

```python
import requests
import pandas as pd

def fetch_bybit_klines(symbol="BTCUSDT", interval="5", limit=1000):
    """Fetch OHLCV data from Bybit V5 API."""
    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    response = requests.get(url, params=params)
    data = response.json()["result"]["list"]

    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume", "turnover"
    ])
    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        df[col] = df[col].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df
```

### Crypto Phase Space Features

```
Crypto-Specific Phase Space:

  q (generalized coordinates):
    q1 = log(close) - EMA(log(close), 20)     # price deviation
    q2 = log(volume) - EMA(log(volume), 20)    # volume deviation
    q3 = (high - low) / close                   # normalized range

  p (conjugate momenta):
    p1 = returns_5m                              # short-term momentum
    p2 = volume_delta / avg_volume               # volume momentum
    p3 = d(volatility)/dt                        # volatility acceleration

  External inputs u(t):
    u1 = funding_rate                            # perpetual futures cost
    u2 = open_interest_change                    # positioning flow
    u3 = liquidation_volume                      # forced selling/buying
```

---

## Training Procedure

### Data Preparation

```python
def prepare_training_data(df, q_cols, p_cols, dt=1.0):
    """Prepare (q, p, dq/dt, dp/dt) training samples."""
    q = df[q_cols].values
    p = df[p_cols].values

    # Compute time derivatives via finite differences
    dq_dt = np.gradient(q, dt, axis=0)
    dp_dt = np.gradient(p, dt, axis=0)

    # Normalize
    q_mean, q_std = q.mean(0), q.std(0)
    p_mean, p_std = p.mean(0), p.std(0)

    q_norm = (q - q_mean) / (q_std + 1e-8)
    p_norm = (p - p_mean) / (p_std + 1e-8)
    dq_dt_norm = dq_dt / (q_std + 1e-8)
    dp_dt_norm = dp_dt / (p_std + 1e-8)

    return q_norm, p_norm, dq_dt_norm, dp_dt_norm
```

### Training Loop

```python
def train_hnn(model, q, p, dq_dt, dp_dt, epochs=500, lr=1e-3):
    """Train HNN on phase-space data."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    q_t = torch.FloatTensor(q).requires_grad_(True)
    p_t = torch.FloatTensor(p).requires_grad_(True)
    dq_target = torch.FloatTensor(dq_dt)
    dp_target = torch.FloatTensor(dp_dt)

    for epoch in range(epochs):
        optimizer.zero_grad()

        dq_pred, dp_pred = model.time_derivative(q_t, p_t)

        loss_q = ((dq_pred - dq_target) ** 2).mean()
        loss_p = ((dp_pred - dp_target) ** 2).mean()
        loss = loss_q + loss_p

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if epoch % 50 == 0:
            H = model.forward(q_t, p_t)
            energy_std = H.std().item()
            print(f"Epoch {epoch}: loss={loss.item():.6f}, "
                  f"energy_std={energy_std:.6f}")
```

### Energy Conservation Monitoring

A well-trained HNN should produce trajectories with approximately constant energy:

```python
def check_energy_conservation(model, q0, p0, dt=0.01, n_steps=1000):
    """Verify that the learned Hamiltonian is conserved."""
    traj_q, traj_p = integrate_trajectory(model, q0, p0, dt, n_steps)

    energies = []
    for i in range(len(traj_q)):
        H = model.forward(traj_q[i:i+1], traj_p[i:i+1])
        energies.append(H.item())

    energies = np.array(energies)
    drift = (energies[-1] - energies[0]) / energies[0]
    oscillation = energies.std() / abs(energies.mean())

    print(f"Energy drift: {drift:.6f}")
    print(f"Energy oscillation: {oscillation:.6f}")
    return energies
```

---

## Trading Strategy

### HNN-Based Trading Signals

The trading strategy uses the HNN to predict future phase-space trajectories and generate signals:

```
Signal Generation Pipeline:

1. Observe current state (q_now, p_now)
2. Integrate forward N steps using leapfrog
3. Predict future price trajectory q(t+1), ..., q(t+N)
4. Compute expected price change: delta_q = q(t+N) - q(t)
5. Compute expected momentum: p(t+N)
6. Generate signal:
   - BUY  if delta_q > threshold AND p(t+N) > 0
   - SELL if delta_q < -threshold AND p(t+N) < 0
   - HOLD otherwise

7. Position sizing: proportional to |delta_q| (confidence)
8. Risk management: stop-loss when energy deviates > 2 sigma
```

### Energy-Based Risk Management

A unique advantage of HNNs is energy-based risk monitoring:

```python
def compute_energy_anomaly(model, q, p, energy_history):
    """Detect regime changes via energy anomalies."""
    H = model.forward(q, p).item()
    mean_H = np.mean(energy_history)
    std_H = np.std(energy_history)

    z_score = (H - mean_H) / (std_H + 1e-8)

    if abs(z_score) > 2.0:
        return "REGIME_CHANGE", z_score
    return "NORMAL", z_score
```

When the system's energy suddenly jumps (breaking conservation), it signals a regime change -- exactly when you should reduce position sizes or exit.

### Backtesting Framework

```python
def backtest_hnn_strategy(model, df, q_cols, p_cols,
                          prediction_horizon=10,
                          threshold=0.001,
                          initial_capital=100000):
    """Backtest HNN trading strategy."""
    capital = initial_capital
    position = 0
    trades = []

    for i in range(len(df) - prediction_horizon):
        q = torch.FloatTensor(df[q_cols].iloc[i].values).unsqueeze(0)
        p = torch.FloatTensor(df[p_cols].iloc[i].values).unsqueeze(0)

        # Predict forward
        traj_q, traj_p = integrate_trajectory(
            model, q, p, dt=1.0, n_steps=prediction_horizon
        )

        predicted_change = (traj_q[-1] - traj_q[0]).item()
        current_price = df["close"].iloc[i]

        # Generate signal
        if predicted_change > threshold and position <= 0:
            position = capital / current_price
            trades.append(("BUY", i, current_price))
        elif predicted_change < -threshold and position >= 0:
            if position > 0:
                capital = position * current_price
            position = -capital / current_price
            trades.append(("SELL", i, current_price))

    return trades, capital
```

---

## Advanced Topics

### Multi-Dimensional Hamiltonian Systems

For multi-asset trading, the Hamiltonian extends to multiple (q, p) pairs:

```
H(q1, q2, ..., qN, p1, p2, ..., pN)

where:
  qi = deviation of asset i from its equilibrium
  pi = momentum of asset i

Cross-asset interactions appear naturally:
  dpi/dt = -dH/dqi  (includes coupling terms d^2H/dqi*dqj)
```

This naturally models cross-asset correlations and contagion effects.

### Separable vs Non-Separable Hamiltonians

**Separable:** H(q, p) = T(p) + V(q)
- Kinetic and potential energy are independent
- Simpler to learn and integrate
- Corresponds to momentum and mean-reversion acting independently

**Non-Separable:** H(q, p) = general function of (q, p)
- Position-dependent effective mass (volatility-dependent momentum)
- More expressive but harder to train
- Better for complex market regimes

### Canonical Transformations for Feature Engineering

Hamiltonian mechanics is invariant under canonical transformations (Q, P) = f(q, p). This means we can choose the "best" coordinates for learning:

```python
def canonical_transform(q, p, transform_type="log"):
    """Apply canonical transformation to phase space."""
    if transform_type == "log":
        # Log transform preserves Hamiltonian structure
        Q = np.log(np.abs(q) + 1e-8) * np.sign(q)
        P = p * np.exp(-Q)  # ensures QP = qp (canonical)
    elif transform_type == "action_angle":
        # Action-angle for near-harmonic systems
        r = np.sqrt(q**2 + p**2)
        theta = np.arctan2(p, q)
        Q = theta  # angle
        P = 0.5 * r**2  # action (~ energy)
    return Q, P
```

---

## Implementation Notes

### Numerical Stability

1. **Gradient clipping** -- autograd through Hamilton's equations can produce large gradients
2. **Normalization** -- q and p should be normalized to similar scales
3. **Learning rate warmup** -- start with small lr, increase gradually
4. **Softplus activation** -- for dissipation network, ensures non-negativity
5. **Weight initialization** -- smaller weights help initial energy estimates

### Hyperparameters

| Parameter | Recommended Range | Notes |
|-----------|------------------|-------|
| Hidden dim | 32-128 | Larger for multi-asset |
| Num layers | 2-4 | More layers for complex H |
| Learning rate | 1e-4 to 1e-3 | Use cosine annealing |
| Integration dt | 0.01-0.1 | Smaller for higher frequency |
| Prediction horizon | 5-50 steps | Asset-dependent |
| Dissipation weight | 0.01-0.1 | Higher for illiquid markets |

### Common Pitfalls

1. **Non-smooth activations** -- ReLU creates non-differentiable points in Hamilton's equations; use tanh or softplus
2. **Ignoring dissipation** -- pure HNNs overpredict oscillation amplitude in real markets
3. **Wrong phase-space coordinates** -- garbage in, garbage out; choose (q, p) carefully
4. **Too large integration steps** -- even symplectic integrators fail with large dt
5. **Overfitting the Hamiltonian** -- regularize H_theta to be smooth

---

## Project Structure

```
149_hamiltonian_nn_trading/
├── README.md                          # This file
├── README.ru.md                       # Russian translation
├── readme.simple.md                   # Simplified explanation (English)
├── readme.simple.ru.md                # Simplified explanation (Russian)
│
├── python/
│   ├── __init__.py
│   ├── requirements.txt
│   ├── hamiltonian_nn.py              # Core HNN model
│   ├── dissipative_hnn.py            # Dissipative HNN extension
│   ├── symplectic_integrator.py       # Leapfrog integrator
│   ├── data_loader.py                 # Bybit + stock data loading
│   ├── train.py                       # Training pipeline
│   ├── visualize.py                   # Phase portraits & energy plots
│   └── backtest.py                    # Trading strategy backtester
│
└── rust_hamiltonian_nn/
    ├── Cargo.toml
    ├── src/
    │   ├── lib.rs                     # Core HNN library
    │   └── bin/
    │       ├── train.rs               # Training binary
    │       ├── predict.rs             # Prediction binary
    │       └── fetch_data.rs          # Bybit data fetcher
    └── examples/
        └── phase_portrait.rs          # Phase portrait visualization
```

## Quick Start

### Python

```bash
cd python
pip install -r requirements.txt

# Fetch data from Bybit
python data_loader.py --symbol BTCUSDT --interval 5 --limit 5000

# Train HNN model
python train.py --model hnn --epochs 500 --lr 1e-3

# Train Dissipative HNN
python train.py --model dissipative --epochs 500

# Visualize phase space
python visualize.py --model saved_model.pt

# Run backtest
python backtest.py --model saved_model.pt --initial-capital 100000
```

### Rust

```bash
cd rust_hamiltonian_nn

# Fetch Bybit data
cargo run --bin fetch_data -- --symbol BTCUSDT --interval 5

# Train model
cargo run --release --bin train -- --epochs 500 --lr 0.001

# Run predictions
cargo run --release --bin predict -- --model model.bin --horizon 20
```

## References

1. Greydanus, S., Dzamba, M., & Sprague, J. (2019). "Hamiltonian Neural Networks." NeurIPS 2019.
2. Zhong, Y.D., Dey, B., & Chakraborty, A. (2020). "Dissipative SymODEN: Encoding Hamiltonian Dynamics with Dissipation and Control into Deep Learning."
3. Cranmer, M., Greydanus, S., Hoyer, S., Battaglia, P., Spergel, D., & Ho, S. (2020). "Lagrangian Neural Networks." ICLR 2020 Workshop.
4. Chen, R.T.Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). "Neural Ordinary Differential Equations." NeurIPS 2018.
5. Hairer, E., Lubich, C., & Wanner, G. (2006). "Geometric Numerical Integration: Structure-Preserving Algorithms for Ordinary Differential Equations."
6. Desmond, S. & Sprague, J. (2020). "Hamiltonian Neural Networks for Solving Equations of Motion."
7. Toth, P., Rezende, D.J., Jaegle, A., Racaniere, S., Botev, A., & Higgins, I. (2020). "Hamiltonian Generative Networks." ICLR 2020.

---

*Chapter 149 of Machine Learning for Trading. For the complete codebase, visit [github.com/suenot/machine-learning-for-trading](https://github.com/suenot/machine-learning-for-trading).*
