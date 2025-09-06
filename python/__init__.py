"""
Hamiltonian Neural Networks for Trading.

This package implements Hamiltonian Neural Networks (HNNs) and their
dissipative extensions for financial market modeling and trading.
The core idea is to model price-momentum dynamics as a Hamiltonian system,
leveraging energy conservation as an inductive bias for stable long-horizon
predictions.

Modules:
    hamiltonian_nn: Core HNN model
    dissipative_hnn: Dissipative HNN with friction modeling
    symplectic_integrator: Leapfrog/Stormer-Verlet integrator
    data_loader: Bybit + stock data loading and phase space construction
    train: Training pipeline
    visualize: Phase portraits and energy plots
    backtest: Trading strategy backtester
"""

__version__ = "0.1.0"
__author__ = "ML4Trading Community"
