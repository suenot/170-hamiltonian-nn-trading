//! # Hamiltonian Neural Networks for Trading
//!
//! This library implements Hamiltonian Neural Networks (HNNs) and their
//! dissipative extensions for financial market modeling and trading.
//!
//! The core idea: learn the Hamiltonian H(q, p) from market data, then derive
//! dynamics via Hamilton's equations using automatic differentiation:
//!   dq/dt =  dH/dp
//!   dp/dt = -dH/dq
//!
//! ## Modules
//! - `nn`: Neural network layers and HNN architecture
//! - `integrator`: Symplectic integrators (leapfrog)
//! - `data`: Bybit API data fetching and phase space construction
//! - `trading`: Trading strategy and backtesting
//! - `utils`: Normalization, metrics, serialization

pub mod nn;
pub mod integrator;
pub mod data;
pub mod trading;
pub mod utils;

pub use nn::{HamiltonianNN, DissipativeHNN};
pub use integrator::{leapfrog_step, integrate_trajectory};
pub use data::{BybitClient, PhaseSpaceData};
pub use trading::{TradingStrategy, BacktestResult};

/// Module: Neural network layers and HNN architecture
pub mod nn {
    use ndarray::{Array1, Array2, Axis};
    use rand::Rng;
    use rand_distr::Normal;
    use serde::{Deserialize, Serialize};

    /// Activation functions (must be smooth for Hamilton's equations)
    #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
    pub enum Activation {
        Tanh,
        Sigmoid,
        Softplus,
    }

    impl Activation {
        pub fn apply(&self, x: f64) -> f64 {
            match self {
                Activation::Tanh => x.tanh(),
                Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
                Activation::Softplus => (1.0 + x.exp()).ln(),
            }
        }

        pub fn derivative(&self, x: f64) -> f64 {
            match self {
                Activation::Tanh => {
                    let t = x.tanh();
                    1.0 - t * t
                }
                Activation::Sigmoid => {
                    let s = 1.0 / (1.0 + (-x).exp());
                    s * (1.0 - s)
                }
                Activation::Softplus => 1.0 / (1.0 + (-x).exp()),
            }
        }
    }

    /// A single dense (fully connected) layer
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct DenseLayer {
        pub weights: Vec<Vec<f64>>,  // [output_dim][input_dim]
        pub biases: Vec<f64>,        // [output_dim]
        pub input_dim: usize,
        pub output_dim: usize,
    }

    impl DenseLayer {
        /// Create a new dense layer with Xavier initialization
        pub fn new(input_dim: usize, output_dim: usize) -> Self {
            let mut rng = rand::thread_rng();
            let std_dev = (2.0 / (input_dim + output_dim) as f64).sqrt();
            let normal = Normal::new(0.0, std_dev).unwrap();

            let weights = (0..output_dim)
                .map(|_| {
                    (0..input_dim)
                        .map(|_| rng.sample(normal))
                        .collect()
                })
                .collect();

            let biases = vec![0.0; output_dim];

            Self {
                weights,
                biases,
                input_dim,
                output_dim,
            }
        }

        /// Forward pass: output = W * input + b
        pub fn forward(&self, input: &[f64]) -> Vec<f64> {
            assert_eq!(input.len(), self.input_dim);
            let mut output = vec![0.0; self.output_dim];
            for i in 0..self.output_dim {
                let mut sum = self.biases[i];
                for j in 0..self.input_dim {
                    sum += self.weights[i][j] * input[j];
                }
                output[i] = sum;
            }
            output
        }

        /// Forward pass with Jacobian computation (for autograd)
        pub fn forward_with_jacobian(&self, input: &[f64]) -> (Vec<f64>, Vec<Vec<f64>>) {
            let output = self.forward(input);
            // Jacobian: d(output_i) / d(input_j) = weights[i][j]
            let jacobian = self.weights.clone();
            (output, jacobian)
        }

        /// Get all parameters as a flat vector
        pub fn parameters(&self) -> Vec<f64> {
            let mut params = Vec::new();
            for row in &self.weights {
                params.extend(row);
            }
            params.extend(&self.biases);
            params
        }

        /// Set parameters from a flat vector
        pub fn set_parameters(&mut self, params: &[f64]) {
            let mut idx = 0;
            for i in 0..self.output_dim {
                for j in 0..self.input_dim {
                    self.weights[i][j] = params[idx];
                    idx += 1;
                }
            }
            for i in 0..self.output_dim {
                self.biases[i] = params[idx];
                idx += 1;
            }
        }

        /// Number of parameters
        pub fn num_parameters(&self) -> usize {
            self.input_dim * self.output_dim + self.output_dim
        }
    }

    /// Multi-layer perceptron with smooth activations
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct MLP {
        pub layers: Vec<DenseLayer>,
        pub activation: Activation,
    }

    impl MLP {
        pub fn new(
            input_dim: usize,
            hidden_dim: usize,
            output_dim: usize,
            num_hidden_layers: usize,
            activation: Activation,
        ) -> Self {
            let mut layers = Vec::new();

            // Input -> first hidden
            layers.push(DenseLayer::new(input_dim, hidden_dim));

            // Hidden layers
            for _ in 1..num_hidden_layers {
                layers.push(DenseLayer::new(hidden_dim, hidden_dim));
            }

            // Last hidden -> output
            layers.push(DenseLayer::new(hidden_dim, output_dim));

            Self { layers, activation }
        }

        /// Forward pass through the MLP
        pub fn forward(&self, input: &[f64]) -> Vec<f64> {
            let mut x = input.to_vec();

            for (i, layer) in self.layers.iter().enumerate() {
                x = layer.forward(&x);
                // Apply activation to all layers except the last
                if i < self.layers.len() - 1 {
                    x = x.iter().map(|&v| self.activation.apply(v)).collect();
                }
            }

            x
        }

        /// Forward pass with tracking of pre-activation values (for gradient computation)
        pub fn forward_with_intermediates(&self, input: &[f64]) -> (Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
            let mut x = input.to_vec();
            let mut pre_activations = Vec::new();
            let mut post_activations = Vec::new();

            post_activations.push(x.clone());

            for (i, layer) in self.layers.iter().enumerate() {
                let z = layer.forward(&x);
                pre_activations.push(z.clone());

                if i < self.layers.len() - 1 {
                    x = z.iter().map(|&v| self.activation.apply(v)).collect();
                } else {
                    x = z;
                }
                post_activations.push(x.clone());
            }

            (x, pre_activations, post_activations)
        }

        /// Compute gradient of scalar output w.r.t. input using backpropagation
        pub fn gradient_wrt_input(&self, input: &[f64]) -> Vec<f64> {
            let (_, pre_activations, post_activations) = self.forward_with_intermediates(input);

            // Start with gradient of output w.r.t. itself = 1
            let output_dim = self.layers.last().unwrap().output_dim;
            let mut grad = vec![1.0; output_dim];

            // Backpropagate through layers
            for i in (0..self.layers.len()).rev() {
                let layer = &self.layers[i];

                // If not last layer, multiply by activation derivative
                if i < self.layers.len() - 1 {
                    let pre_act = &pre_activations[i];
                    for j in 0..grad.len() {
                        grad[j] *= self.activation.derivative(pre_act[j]);
                    }
                }

                // Multiply by transpose of weight matrix
                let mut new_grad = vec![0.0; layer.input_dim];
                for j in 0..layer.input_dim {
                    for k in 0..layer.output_dim {
                        new_grad[j] += layer.weights[k][j] * grad[k];
                    }
                }
                grad = new_grad;
            }

            grad
        }

        /// Get all parameters
        pub fn parameters(&self) -> Vec<f64> {
            let mut params = Vec::new();
            for layer in &self.layers {
                params.extend(layer.parameters());
            }
            params
        }

        /// Set parameters
        pub fn set_parameters(&mut self, params: &[f64]) {
            let mut idx = 0;
            for layer in &mut self.layers {
                let n = layer.num_parameters();
                layer.set_parameters(&params[idx..idx + n]);
                idx += n;
            }
        }

        /// Total number of parameters
        pub fn num_parameters(&self) -> usize {
            self.layers.iter().map(|l| l.num_parameters()).sum()
        }
    }

    /// Hamiltonian Neural Network
    ///
    /// Learns H(q, p) as a scalar function. Dynamics derived via:
    ///   dq/dt =  dH/dp
    ///   dp/dt = -dH/dq
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct HamiltonianNN {
        pub h_net: MLP,
        pub coord_dim: usize,
    }

    impl HamiltonianNN {
        pub fn new(
            coord_dim: usize,
            hidden_dim: usize,
            num_layers: usize,
        ) -> Self {
            let input_dim = 2 * coord_dim;
            let h_net = MLP::new(input_dim, hidden_dim, 1, num_layers, Activation::Tanh);
            Self { h_net, coord_dim }
        }

        /// Compute the Hamiltonian H(q, p)
        pub fn hamiltonian(&self, q: &[f64], p: &[f64]) -> f64 {
            let mut input = Vec::with_capacity(q.len() + p.len());
            input.extend_from_slice(q);
            input.extend_from_slice(p);
            self.h_net.forward(&input)[0]
        }

        /// Compute Hamilton's equations: (dq/dt, dp/dt)
        ///
        /// dq/dt =  dH/dp
        /// dp/dt = -dH/dq
        pub fn time_derivative(&self, q: &[f64], p: &[f64]) -> (Vec<f64>, Vec<f64>) {
            let mut input = Vec::with_capacity(q.len() + p.len());
            input.extend_from_slice(q);
            input.extend_from_slice(p);

            let grad = self.h_net.gradient_wrt_input(&input);

            let dh_dq: Vec<f64> = grad[..self.coord_dim].to_vec();
            let dh_dp: Vec<f64> = grad[self.coord_dim..].to_vec();

            // Hamilton's equations
            let dq_dt = dh_dp;  // dq/dt = dH/dp
            let dp_dt: Vec<f64> = dh_dq.iter().map(|&x| -x).collect();  // dp/dt = -dH/dq

            (dq_dt, dp_dt)
        }

        pub fn parameters(&self) -> Vec<f64> {
            self.h_net.parameters()
        }

        pub fn set_parameters(&mut self, params: &[f64]) {
            self.h_net.set_parameters(params);
        }

        pub fn num_parameters(&self) -> usize {
            self.h_net.num_parameters()
        }
    }

    /// Dissipative Hamiltonian Neural Network
    ///
    /// Extends HNN with a dissipation function D(q, p) >= 0:
    ///   dq/dt =  dH/dp
    ///   dp/dt = -dH/dq - dD/dp
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct DissipativeHNN {
        pub h_net: MLP,
        pub d_net: MLP,
        pub coord_dim: usize,
    }

    impl DissipativeHNN {
        pub fn new(
            coord_dim: usize,
            hidden_dim: usize,
            num_layers: usize,
        ) -> Self {
            let input_dim = 2 * coord_dim;
            let h_net = MLP::new(input_dim, hidden_dim, 1, num_layers, Activation::Tanh);
            let d_net = MLP::new(input_dim, hidden_dim, 1, num_layers, Activation::Tanh);
            Self { h_net, d_net, coord_dim }
        }

        pub fn hamiltonian(&self, q: &[f64], p: &[f64]) -> f64 {
            let mut input = Vec::with_capacity(q.len() + p.len());
            input.extend_from_slice(q);
            input.extend_from_slice(p);
            self.h_net.forward(&input)[0]
        }

        /// Dissipation function (always non-negative via softplus on output)
        pub fn dissipation(&self, q: &[f64], p: &[f64]) -> f64 {
            let mut input = Vec::with_capacity(q.len() + p.len());
            input.extend_from_slice(q);
            input.extend_from_slice(p);
            let raw = self.d_net.forward(&input)[0];
            // Softplus to ensure non-negativity
            (1.0 + raw.exp()).ln()
        }

        /// Compute dissipative Hamilton's equations
        pub fn time_derivative(&self, q: &[f64], p: &[f64]) -> (Vec<f64>, Vec<f64>) {
            let mut input = Vec::with_capacity(q.len() + p.len());
            input.extend_from_slice(q);
            input.extend_from_slice(p);

            // Hamiltonian gradients
            let h_grad = self.h_net.gradient_wrt_input(&input);
            let dh_dq: Vec<f64> = h_grad[..self.coord_dim].to_vec();
            let dh_dp: Vec<f64> = h_grad[self.coord_dim..].to_vec();

            // Dissipation gradient (need dD/dp)
            let d_grad = self.d_net.gradient_wrt_input(&input);
            let dd_dp: Vec<f64> = d_grad[self.coord_dim..].to_vec();

            // For softplus: d(softplus(x))/dx = sigmoid(x) = dD/d(raw) * d(raw)/dp
            let raw = self.d_net.forward(&input)[0];
            let softplus_deriv = 1.0 / (1.0 + (-raw).exp());
            let dd_dp_corrected: Vec<f64> = dd_dp.iter().map(|&x| x * softplus_deriv).collect();

            // Dissipative Hamilton's equations
            let dq_dt = dh_dp;
            let dp_dt: Vec<f64> = dh_dq
                .iter()
                .zip(dd_dp_corrected.iter())
                .map(|(&dh, &dd)| -dh - dd)
                .collect();

            (dq_dt, dp_dt)
        }

        pub fn parameters(&self) -> Vec<f64> {
            let mut params = self.h_net.parameters();
            params.extend(self.d_net.parameters());
            params
        }

        pub fn set_parameters(&mut self, params: &[f64]) {
            let h_n = self.h_net.num_parameters();
            self.h_net.set_parameters(&params[..h_n]);
            self.d_net.set_parameters(&params[h_n..]);
        }

        pub fn num_parameters(&self) -> usize {
            self.h_net.num_parameters() + self.d_net.num_parameters()
        }
    }
}

/// Module: Symplectic integrators
pub mod integrator {
    use super::nn::{HamiltonianNN, DissipativeHNN};

    /// One step of leapfrog (Stormer-Verlet) symplectic integration for HNN
    pub fn leapfrog_step(
        model: &HamiltonianNN,
        q: &[f64],
        p: &[f64],
        dt: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let dim = q.len();

        // Half-step momentum
        let (_, dp_dt_0) = model.time_derivative(q, p);
        let p_half: Vec<f64> = p.iter()
            .zip(dp_dt_0.iter())
            .map(|(&pi, &dpi)| pi + 0.5 * dt * dpi)
            .collect();

        // Full-step position
        let (dq_dt_half, _) = model.time_derivative(q, &p_half);
        let q_new: Vec<f64> = q.iter()
            .zip(dq_dt_half.iter())
            .map(|(&qi, &dqi)| qi + dt * dqi)
            .collect();

        // Half-step momentum
        let (_, dp_dt_1) = model.time_derivative(&q_new, &p_half);
        let p_new: Vec<f64> = p_half.iter()
            .zip(dp_dt_1.iter())
            .map(|(&pi, &dpi)| pi + 0.5 * dt * dpi)
            .collect();

        (q_new, p_new)
    }

    /// Leapfrog step for dissipative HNN
    pub fn leapfrog_step_dissipative(
        model: &DissipativeHNN,
        q: &[f64],
        p: &[f64],
        dt: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        // Half-step momentum
        let (_, dp_dt_0) = model.time_derivative(q, p);
        let p_half: Vec<f64> = p.iter()
            .zip(dp_dt_0.iter())
            .map(|(&pi, &dpi)| pi + 0.5 * dt * dpi)
            .collect();

        // Full-step position
        let (dq_dt_half, _) = model.time_derivative(q, &p_half);
        let q_new: Vec<f64> = q.iter()
            .zip(dq_dt_half.iter())
            .map(|(&qi, &dqi)| qi + dt * dqi)
            .collect();

        // Half-step momentum
        let (_, dp_dt_1) = model.time_derivative(&q_new, &p_half);
        let p_new: Vec<f64> = p_half.iter()
            .zip(dp_dt_1.iter())
            .map(|(&pi, &dpi)| pi + 0.5 * dt * dpi)
            .collect();

        (q_new, p_new)
    }

    /// Euler integration (non-symplectic, for comparison)
    pub fn euler_step(
        model: &HamiltonianNN,
        q: &[f64],
        p: &[f64],
        dt: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let (dq_dt, dp_dt) = model.time_derivative(q, p);
        let q_new: Vec<f64> = q.iter()
            .zip(dq_dt.iter())
            .map(|(&qi, &dqi)| qi + dt * dqi)
            .collect();
        let p_new: Vec<f64> = p.iter()
            .zip(dp_dt.iter())
            .map(|(&pi, &dpi)| pi + dt * dpi)
            .collect();
        (q_new, p_new)
    }

    /// Integrate a trajectory using leapfrog
    pub fn integrate_trajectory(
        model: &HamiltonianNN,
        q0: &[f64],
        p0: &[f64],
        dt: f64,
        n_steps: usize,
    ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let mut traj_q = vec![q0.to_vec()];
        let mut traj_p = vec![p0.to_vec()];

        let mut q = q0.to_vec();
        let mut p = p0.to_vec();

        for _ in 0..n_steps {
            let (q_new, p_new) = leapfrog_step(model, &q, &p, dt);
            traj_q.push(q_new.clone());
            traj_p.push(p_new.clone());
            q = q_new;
            p = p_new;
        }

        (traj_q, traj_p)
    }

    /// Integrate with dissipative model
    pub fn integrate_trajectory_dissipative(
        model: &DissipativeHNN,
        q0: &[f64],
        p0: &[f64],
        dt: f64,
        n_steps: usize,
    ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let mut traj_q = vec![q0.to_vec()];
        let mut traj_p = vec![p0.to_vec()];

        let mut q = q0.to_vec();
        let mut p = p0.to_vec();

        for _ in 0..n_steps {
            let (q_new, p_new) = leapfrog_step_dissipative(model, &q, &p, dt);
            traj_q.push(q_new.clone());
            traj_p.push(p_new.clone());
            q = q_new;
            p = p_new;
        }

        (traj_q, traj_p)
    }
}

/// Module: Data fetching and phase space construction
pub mod data {
    use serde::{Deserialize, Serialize};
    use anyhow::Result;

    /// OHLCV candle data
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Candle {
        pub timestamp: i64,
        pub open: f64,
        pub high: f64,
        pub low: f64,
        pub close: f64,
        pub volume: f64,
        pub turnover: f64,
    }

    /// Phase space data
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct PhaseSpaceData {
        pub q: Vec<Vec<f64>>,
        pub p: Vec<Vec<f64>>,
        pub dq_dt: Vec<Vec<f64>>,
        pub dp_dt: Vec<Vec<f64>>,
        pub prices: Vec<f64>,
        pub timestamps: Vec<i64>,
    }

    /// Normalization statistics
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct NormStats {
        pub q_mean: Vec<f64>,
        pub q_std: Vec<f64>,
        pub p_mean: Vec<f64>,
        pub p_std: Vec<f64>,
    }

    /// Bybit API response structures
    #[derive(Debug, Deserialize)]
    pub struct BybitResponse {
        #[serde(rename = "retCode")]
        pub ret_code: i32,
        #[serde(rename = "retMsg")]
        pub ret_msg: String,
        pub result: BybitResult,
    }

    #[derive(Debug, Deserialize)]
    pub struct BybitResult {
        pub list: Vec<Vec<String>>,
    }

    /// Bybit API client
    pub struct BybitClient {
        pub base_url: String,
        client: reqwest::Client,
    }

    impl BybitClient {
        pub fn new() -> Self {
            Self {
                base_url: "https://api.bybit.com".to_string(),
                client: reqwest::Client::new(),
            }
        }

        /// Fetch kline data from Bybit V5 API
        pub async fn fetch_klines(
            &self,
            symbol: &str,
            interval: &str,
            limit: usize,
            end_time: Option<i64>,
        ) -> Result<Vec<Candle>> {
            let mut url = format!(
                "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
                self.base_url, symbol, interval, limit.min(1000)
            );

            if let Some(et) = end_time {
                url.push_str(&format!("&end={}", et));
            }

            let resp: BybitResponse = self.client
                .get(&url)
                .send()
                .await?
                .json()
                .await?;

            if resp.ret_code != 0 {
                anyhow::bail!("Bybit API error: {}", resp.ret_msg);
            }

            let mut candles: Vec<Candle> = resp.result.list
                .iter()
                .map(|row| {
                    Candle {
                        timestamp: row[0].parse().unwrap_or(0),
                        open: row[1].parse().unwrap_or(0.0),
                        high: row[2].parse().unwrap_or(0.0),
                        low: row[3].parse().unwrap_or(0.0),
                        close: row[4].parse().unwrap_or(0.0),
                        volume: row[5].parse().unwrap_or(0.0),
                        turnover: row.get(6)
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(0.0),
                    }
                })
                .collect();

            candles.sort_by_key(|c| c.timestamp);
            Ok(candles)
        }

        /// Fetch extended history by paginating
        pub async fn fetch_extended(
            &self,
            symbol: &str,
            interval: &str,
            total_candles: usize,
        ) -> Result<Vec<Candle>> {
            let mut all_candles = Vec::new();
            let mut end_time: Option<i64> = None;
            let mut remaining = total_candles;

            while remaining > 0 {
                let batch_size = remaining.min(1000);
                let candles = self.fetch_klines(symbol, interval, batch_size, end_time).await?;

                if candles.is_empty() {
                    break;
                }

                let earliest = candles.first().unwrap().timestamp;
                end_time = Some(earliest - 1);
                remaining = remaining.saturating_sub(candles.len());

                all_candles.extend(candles);
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            }

            all_candles.sort_by_key(|c| c.timestamp);
            all_candles.dedup_by_key(|c| c.timestamp);
            Ok(all_candles)
        }
    }

    /// Construct phase space from candle data
    pub fn construct_phase_space(
        candles: &[Candle],
        ma_window: usize,
    ) -> PhaseSpaceData {
        let n = candles.len();
        let log_close: Vec<f64> = candles.iter().map(|c| c.close.ln()).collect();

        // Moving average
        let mut ma = vec![f64::NAN; n];
        for i in (ma_window - 1)..n {
            let sum: f64 = log_close[i + 1 - ma_window..=i].iter().sum();
            ma[i] = sum / ma_window as f64;
        }

        // q: price deviation from MA
        let q_raw: Vec<f64> = (0..n)
            .map(|i| log_close[i] - ma[i])
            .collect();

        // p: log returns (gradient approximation)
        let mut p_raw = vec![0.0; n];
        for i in 1..n {
            p_raw[i] = log_close[i] - log_close[i - 1];
        }

        // Time derivatives
        let mut dq_dt = vec![0.0; n];
        let mut dp_dt = vec![0.0; n];
        for i in 1..n - 1 {
            dq_dt[i] = (q_raw[i + 1] - q_raw[i - 1]) / 2.0;
            dp_dt[i] = (p_raw[i + 1] - p_raw[i - 1]) / 2.0;
        }

        // Filter valid (non-NaN) entries
        let mut ps_data = PhaseSpaceData {
            q: Vec::new(),
            p: Vec::new(),
            dq_dt: Vec::new(),
            dp_dt: Vec::new(),
            prices: Vec::new(),
            timestamps: Vec::new(),
        };

        for i in ma_window..n - 1 {
            if q_raw[i].is_finite() && p_raw[i].is_finite()
                && dq_dt[i].is_finite() && dp_dt[i].is_finite()
            {
                ps_data.q.push(vec![q_raw[i]]);
                ps_data.p.push(vec![p_raw[i]]);
                ps_data.dq_dt.push(vec![dq_dt[i]]);
                ps_data.dp_dt.push(vec![dp_dt[i]]);
                ps_data.prices.push(candles[i].close);
                ps_data.timestamps.push(candles[i].timestamp);
            }
        }

        ps_data
    }

    /// Normalize phase space data
    pub fn normalize_phase_space(data: &PhaseSpaceData) -> (PhaseSpaceData, NormStats) {
        let n = data.q.len();
        let dim = if n > 0 { data.q[0].len() } else { 1 };

        let mut q_mean = vec![0.0; dim];
        let mut p_mean = vec![0.0; dim];

        for i in 0..n {
            for d in 0..dim {
                q_mean[d] += data.q[i][d];
                p_mean[d] += data.p[i][d];
            }
        }
        for d in 0..dim {
            q_mean[d] /= n as f64;
            p_mean[d] /= n as f64;
        }

        let mut q_var = vec![0.0; dim];
        let mut p_var = vec![0.0; dim];
        for i in 0..n {
            for d in 0..dim {
                q_var[d] += (data.q[i][d] - q_mean[d]).powi(2);
                p_var[d] += (data.p[i][d] - p_mean[d]).powi(2);
            }
        }
        let q_std: Vec<f64> = q_var.iter().map(|&v| (v / n as f64).sqrt().max(1e-8)).collect();
        let p_std: Vec<f64> = p_var.iter().map(|&v| (v / n as f64).sqrt().max(1e-8)).collect();

        let stats = NormStats {
            q_mean: q_mean.clone(),
            q_std: q_std.clone(),
            p_mean: p_mean.clone(),
            p_std: p_std.clone(),
        };

        let mut normalized = PhaseSpaceData {
            q: Vec::with_capacity(n),
            p: Vec::with_capacity(n),
            dq_dt: Vec::with_capacity(n),
            dp_dt: Vec::with_capacity(n),
            prices: data.prices.clone(),
            timestamps: data.timestamps.clone(),
        };

        for i in 0..n {
            let q_norm: Vec<f64> = (0..dim)
                .map(|d| (data.q[i][d] - q_mean[d]) / q_std[d])
                .collect();
            let p_norm: Vec<f64> = (0..dim)
                .map(|d| (data.p[i][d] - p_mean[d]) / p_std[d])
                .collect();
            let dq_norm: Vec<f64> = (0..dim)
                .map(|d| data.dq_dt[i][d] / q_std[d])
                .collect();
            let dp_norm: Vec<f64> = (0..dim)
                .map(|d| data.dp_dt[i][d] / p_std[d])
                .collect();

            normalized.q.push(q_norm);
            normalized.p.push(p_norm);
            normalized.dq_dt.push(dq_norm);
            normalized.dp_dt.push(dp_norm);
        }

        (normalized, stats)
    }
}

/// Module: Trading strategy and backtesting
pub mod trading {
    use super::nn::HamiltonianNN;
    use super::integrator;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Trade {
        pub timestamp_idx: usize,
        pub side: String,
        pub price: f64,
        pub quantity: f64,
        pub energy: f64,
        pub energy_zscore: f64,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct BacktestResult {
        pub initial_capital: f64,
        pub final_capital: f64,
        pub total_return: f64,
        pub max_drawdown: f64,
        pub sharpe_ratio: f64,
        pub win_rate: f64,
        pub n_trades: usize,
        pub trades: Vec<Trade>,
        pub equity_curve: Vec<f64>,
    }

    pub struct TradingStrategy {
        pub model: HamiltonianNN,
        pub prediction_horizon: usize,
        pub dt: f64,
        pub entry_threshold: f64,
        pub stop_loss_pct: f64,
        pub take_profit_pct: f64,
        pub energy_history: Vec<f64>,
    }

    impl TradingStrategy {
        pub fn new(
            model: HamiltonianNN,
            prediction_horizon: usize,
            dt: f64,
            entry_threshold: f64,
        ) -> Self {
            Self {
                model,
                prediction_horizon,
                dt,
                entry_threshold,
                stop_loss_pct: 0.03,
                take_profit_pct: 0.05,
                energy_history: Vec::new(),
            }
        }

        /// Generate a trading signal
        pub fn generate_signal(&mut self, q: &[f64], p: &[f64]) -> (String, f64, f64) {
            let energy = self.model.hamiltonian(q, p);
            self.energy_history.push(energy);

            // Integrate forward
            let (traj_q, _traj_p) = integrator::integrate_trajectory(
                &self.model, q, p, self.dt, self.prediction_horizon,
            );

            let predicted_change = traj_q.last().unwrap()[0] - traj_q[0][0];
            let strength = predicted_change.abs();

            // Energy z-score for regime detection
            let zscore = if self.energy_history.len() >= 20 {
                let recent: Vec<f64> = self.energy_history
                    .iter()
                    .rev()
                    .take(100)
                    .copied()
                    .collect();
                let mean: f64 = recent.iter().sum::<f64>() / recent.len() as f64;
                let var: f64 = recent.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                    / recent.len() as f64;
                let std = var.sqrt().max(1e-10);
                (energy - mean) / std
            } else {
                0.0
            };

            // Regime change detection
            if zscore.abs() > 2.5 {
                return ("HOLD".to_string(), strength, zscore);
            }

            if strength > self.entry_threshold {
                if predicted_change > 0.0 {
                    return ("BUY".to_string(), strength, zscore);
                } else {
                    return ("SELL".to_string(), strength, zscore);
                }
            }

            ("HOLD".to_string(), strength, zscore)
        }

        /// Run a backtest
        pub fn backtest(
            &mut self,
            prices: &[f64],
            q_data: &[Vec<f64>],
            p_data: &[Vec<f64>],
            initial_capital: f64,
            commission: f64,
        ) -> BacktestResult {
            let n = prices.len().min(q_data.len());

            let mut capital = initial_capital;
            let mut position = 0.0_f64;
            let mut entry_price = 0.0_f64;
            let mut trades = Vec::new();
            let mut equity_curve = Vec::with_capacity(n);

            for i in 0..n {
                let price = prices[i];
                let (signal, strength, zscore) = self.generate_signal(&q_data[i], &p_data[i]);

                // Stop-loss / take-profit
                if position.abs() > 1e-10 {
                    let pnl_pct = if position > 0.0 {
                        (price - entry_price) / entry_price
                    } else {
                        (entry_price - price) / entry_price
                    };

                    if pnl_pct <= -self.stop_loss_pct || pnl_pct >= self.take_profit_pct {
                        let pnl = if position > 0.0 {
                            position * (price - entry_price)
                        } else {
                            -position * (entry_price - price)
                        };
                        capital += pnl - position.abs() * price * commission;
                        trades.push(Trade {
                            timestamp_idx: i,
                            side: "CLOSE".to_string(),
                            price,
                            quantity: position.abs(),
                            energy: self.model.hamiltonian(&q_data[i], &p_data[i]),
                            energy_zscore: zscore,
                        });
                        position = 0.0;
                    }
                }

                // Execute signal
                match signal.as_str() {
                    "BUY" if position <= 0.0 => {
                        if position < 0.0 {
                            let pnl = -position * (entry_price - price);
                            capital += pnl - position.abs() * price * commission;
                        }
                        let qty = capital / price;
                        position = qty;
                        entry_price = price;
                        capital -= qty * price * commission;
                        trades.push(Trade {
                            timestamp_idx: i,
                            side: "BUY".to_string(),
                            price,
                            quantity: qty,
                            energy: self.model.hamiltonian(&q_data[i], &p_data[i]),
                            energy_zscore: zscore,
                        });
                    }
                    "SELL" if position >= 0.0 => {
                        if position > 0.0 {
                            let pnl = position * (price - entry_price);
                            capital += pnl - position * price * commission;
                        }
                        let qty = capital / price;
                        position = -qty;
                        entry_price = price;
                        capital -= qty * price * commission;
                        trades.push(Trade {
                            timestamp_idx: i,
                            side: "SELL".to_string(),
                            price,
                            quantity: qty,
                            energy: self.model.hamiltonian(&q_data[i], &p_data[i]),
                            energy_zscore: zscore,
                        });
                    }
                    _ => {}
                }

                // Update equity
                let equity = if position > 0.0 {
                    capital + position * (price - entry_price)
                } else if position < 0.0 {
                    capital - position * (price - entry_price)
                } else {
                    capital
                };
                equity_curve.push(equity);
            }

            // Close remaining position
            if position.abs() > 1e-10 {
                let final_price = prices[n - 1];
                let pnl = if position > 0.0 {
                    position * (final_price - entry_price)
                } else {
                    -position * (entry_price - final_price)
                };
                capital += pnl;
            }

            // Compute metrics
            let total_return = (capital - initial_capital) / initial_capital;

            let mut peak = f64::MIN;
            let mut max_dd = 0.0_f64;
            for &eq in &equity_curve {
                if eq > peak { peak = eq; }
                let dd = (eq - peak) / peak;
                if dd < max_dd { max_dd = dd; }
            }

            let returns: Vec<f64> = equity_curve.windows(2)
                .map(|w| (w[1] - w[0]) / w[0].max(1e-10))
                .collect();
            let mean_ret = returns.iter().sum::<f64>() / returns.len().max(1) as f64;
            let std_ret = {
                let var: f64 = returns.iter()
                    .map(|&r| (r - mean_ret).powi(2))
                    .sum::<f64>() / returns.len().max(1) as f64;
                var.sqrt().max(1e-10)
            };
            let sharpe = mean_ret / std_ret * (252.0 * 288.0_f64).sqrt();

            let win_count = trades.chunks(2)
                .filter(|chunk| {
                    if chunk.len() == 2 {
                        let entry = &chunk[0];
                        let exit = &chunk[1];
                        if entry.side == "BUY" {
                            exit.price > entry.price
                        } else {
                            exit.price < entry.price
                        }
                    } else { false }
                })
                .count();
            let total_pairs = (trades.len() / 2).max(1);
            let win_rate = win_count as f64 / total_pairs as f64;

            BacktestResult {
                initial_capital,
                final_capital: capital,
                total_return,
                max_drawdown: max_dd,
                sharpe_ratio: sharpe,
                win_rate,
                n_trades: trades.len(),
                trades,
                equity_curve,
            }
        }
    }
}

/// Module: Utility functions
pub mod utils {
    use super::nn::HamiltonianNN;

    /// Simple SGD optimizer for training
    pub struct SGDOptimizer {
        pub learning_rate: f64,
        pub momentum: f64,
        velocity: Vec<f64>,
    }

    impl SGDOptimizer {
        pub fn new(n_params: usize, learning_rate: f64, momentum: f64) -> Self {
            Self {
                learning_rate,
                momentum,
                velocity: vec![0.0; n_params],
            }
        }

        pub fn step(&mut self, params: &mut [f64], gradients: &[f64]) {
            for i in 0..params.len() {
                self.velocity[i] = self.momentum * self.velocity[i] - self.learning_rate * gradients[i];
                params[i] += self.velocity[i];
            }
        }
    }

    /// Compute MSE loss for HNN training
    ///
    /// Uses finite-difference approximation for parameter gradients
    pub fn compute_loss_and_gradients(
        model: &HamiltonianNN,
        q_batch: &[Vec<f64>],
        p_batch: &[Vec<f64>],
        dq_target: &[Vec<f64>],
        dp_target: &[Vec<f64>],
    ) -> (f64, Vec<f64>) {
        let batch_size = q_batch.len();
        let dim = q_batch[0].len();

        // Forward pass: compute loss
        let mut total_loss = 0.0;
        for i in 0..batch_size {
            let (dq_pred, dp_pred) = model.time_derivative(&q_batch[i], &p_batch[i]);
            for d in 0..dim {
                total_loss += (dq_pred[d] - dq_target[i][d]).powi(2);
                total_loss += (dp_pred[d] - dp_target[i][d]).powi(2);
            }
        }
        total_loss /= batch_size as f64;

        // Finite-difference gradients for parameters
        let params = model.parameters();
        let n_params = params.len();
        let mut gradients = vec![0.0; n_params];
        let eps = 1e-5;

        for j in 0..n_params {
            let mut params_plus = params.clone();
            params_plus[j] += eps;

            let mut model_plus = model.clone();
            model_plus.set_parameters(&params_plus);

            let mut loss_plus = 0.0;
            for i in 0..batch_size {
                let (dq_pred, dp_pred) = model_plus.time_derivative(&q_batch[i], &p_batch[i]);
                for d in 0..dim {
                    loss_plus += (dq_pred[d] - dq_target[i][d]).powi(2);
                    loss_plus += (dp_pred[d] - dp_target[i][d]).powi(2);
                }
            }
            loss_plus /= batch_size as f64;

            gradients[j] = (loss_plus - total_loss) / eps;
        }

        (total_loss, gradients)
    }

    /// Compute energy along a trajectory
    pub fn energy_along_trajectory(
        model: &HamiltonianNN,
        traj_q: &[Vec<f64>],
        traj_p: &[Vec<f64>],
    ) -> Vec<f64> {
        traj_q.iter()
            .zip(traj_p.iter())
            .map(|(q, p)| model.hamiltonian(q, p))
            .collect()
    }

    /// Export data to CSV
    pub fn export_csv(
        path: &str,
        headers: &[&str],
        data: &[Vec<f64>],
    ) -> std::io::Result<()> {
        let mut wtr = csv::Writer::from_path(path)?;
        wtr.write_record(headers)?;
        for row in data {
            let record: Vec<String> = row.iter().map(|v| format!("{:.8}", v)).collect();
            wtr.write_record(&record)?;
        }
        wtr.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hnn_creation() {
        let model = nn::HamiltonianNN::new(1, 32, 2);
        assert_eq!(model.coord_dim, 1);
        assert!(model.num_parameters() > 0);
    }

    #[test]
    fn test_hnn_forward() {
        let model = nn::HamiltonianNN::new(1, 32, 2);
        let q = vec![0.5];
        let p = vec![-0.3];

        let h = model.hamiltonian(&q, &p);
        assert!(h.is_finite());
    }

    #[test]
    fn test_hnn_time_derivative() {
        let model = nn::HamiltonianNN::new(1, 32, 2);
        let q = vec![1.0];
        let p = vec![0.0];

        let (dq_dt, dp_dt) = model.time_derivative(&q, &p);
        assert_eq!(dq_dt.len(), 1);
        assert_eq!(dp_dt.len(), 1);
        assert!(dq_dt[0].is_finite());
        assert!(dp_dt[0].is_finite());
    }

    #[test]
    fn test_leapfrog_integration() {
        let model = nn::HamiltonianNN::new(1, 32, 2);
        let q0 = vec![1.0];
        let p0 = vec![0.0];

        let (traj_q, traj_p) = integrator::integrate_trajectory(&model, &q0, &p0, 0.01, 100);
        assert_eq!(traj_q.len(), 101);
        assert_eq!(traj_p.len(), 101);

        // Check all values are finite
        for (q, p) in traj_q.iter().zip(traj_p.iter()) {
            assert!(q[0].is_finite(), "q became non-finite during integration");
            assert!(p[0].is_finite(), "p became non-finite during integration");
        }
    }

    #[test]
    fn test_dissipative_hnn() {
        let model = nn::DissipativeHNN::new(1, 32, 2);
        let q = vec![0.5];
        let p = vec![-0.3];

        let h = model.hamiltonian(&q, &p);
        let d = model.dissipation(&q, &p);
        assert!(h.is_finite());
        assert!(d >= 0.0, "Dissipation must be non-negative");

        let (dq_dt, dp_dt) = model.time_derivative(&q, &p);
        assert!(dq_dt[0].is_finite());
        assert!(dp_dt[0].is_finite());
    }

    #[test]
    fn test_dense_layer() {
        let layer = nn::DenseLayer::new(3, 5);
        assert_eq!(layer.input_dim, 3);
        assert_eq!(layer.output_dim, 5);

        let input = vec![1.0, 2.0, 3.0];
        let output = layer.forward(&input);
        assert_eq!(output.len(), 5);
    }

    #[test]
    fn test_mlp() {
        let mlp = nn::MLP::new(2, 16, 1, 2, nn::Activation::Tanh);
        let input = vec![0.5, -0.3];
        let output = mlp.forward(&input);
        assert_eq!(output.len(), 1);
        assert!(output[0].is_finite());
    }

    #[test]
    fn test_gradient_computation() {
        let mlp = nn::MLP::new(2, 8, 1, 2, nn::Activation::Tanh);
        let input = vec![1.0, 0.5];
        let grad = mlp.gradient_wrt_input(&input);
        assert_eq!(grad.len(), 2);
        assert!(grad[0].is_finite());
        assert!(grad[1].is_finite());
    }

    #[test]
    fn test_serialization() {
        let model = nn::HamiltonianNN::new(1, 16, 2);
        let serialized = serde_json::to_string(&model).unwrap();
        let deserialized: nn::HamiltonianNN = serde_json::from_str(&serialized).unwrap();
        assert_eq!(model.coord_dim, deserialized.coord_dim);
        assert_eq!(model.num_parameters(), deserialized.num_parameters());
    }
}
