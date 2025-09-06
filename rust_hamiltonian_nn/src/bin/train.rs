//! Train a Hamiltonian Neural Network on market data.
//!
//! Usage:
//!   cargo run --release --bin train -- --epochs 500 --lr 0.001
//!   cargo run --release --bin train -- --data data/BTCUSDT_phase_space.json --epochs 1000

use anyhow::Result;
use clap::Parser;
use hamiltonian_nn_trading::data::PhaseSpaceData;
use hamiltonian_nn_trading::nn::HamiltonianNN;
use hamiltonian_nn_trading::utils::{compute_loss_and_gradients, SGDOptimizer, energy_along_trajectory};
use hamiltonian_nn_trading::integrator::integrate_trajectory;
use indicatif::{ProgressBar, ProgressStyle};

#[derive(Parser, Debug)]
#[command(name = "train", about = "Train HNN on market data")]
struct Args {
    /// Path to phase space JSON data
    #[arg(long, default_value = "data/BTCUSDT_phase_space.json")]
    data: String,

    /// Number of training epochs
    #[arg(long, default_value_t = 500)]
    epochs: usize,

    /// Learning rate
    #[arg(long, default_value_t = 0.001)]
    lr: f64,

    /// Momentum for SGD
    #[arg(long, default_value_t = 0.9)]
    momentum: f64,

    /// Hidden layer dimension
    #[arg(long, default_value_t = 32)]
    hidden_dim: usize,

    /// Number of hidden layers
    #[arg(long, default_value_t = 2)]
    num_layers: usize,

    /// Batch size for training
    #[arg(long, default_value_t = 64)]
    batch_size: usize,

    /// Output model path
    #[arg(long, default_value = "output/model.json")]
    output: String,

    /// Use synthetic data for testing
    #[arg(long)]
    synthetic: bool,
}

/// Generate synthetic harmonic oscillator data for testing
fn generate_synthetic_data(n_samples: usize) -> PhaseSpaceData {
    let dt = 0.01;
    let mut q_data = Vec::new();
    let mut p_data = Vec::new();
    let mut dq_dt = Vec::new();
    let mut dp_dt = Vec::new();
    let mut prices = Vec::new();

    for i in 0..n_samples {
        let t = i as f64 * dt;
        let q = (t).cos();
        let p = -(t).sin();
        let dq = -(t).sin();
        let dp = -(t).cos();

        q_data.push(vec![q]);
        p_data.push(vec![p]);
        dq_dt.push(vec![dq]);
        dp_dt.push(vec![dp]);
        prices.push(100.0 * (1.0 + 0.01 * q));
    }

    PhaseSpaceData {
        q: q_data,
        p: p_data,
        dq_dt,
        dp_dt,
        prices,
        timestamps: (0..n_samples as i64).collect(),
    }
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    // Load or generate data
    let data = if args.synthetic {
        println!("Using synthetic harmonic oscillator data...");
        generate_synthetic_data(2000)
    } else {
        println!("Loading data from {}...", args.data);
        let json_str = std::fs::read_to_string(&args.data)?;
        let json: serde_json::Value = serde_json::from_str(&json_str)?;

        PhaseSpaceData {
            q: serde_json::from_value(json["q"].clone())?,
            p: serde_json::from_value(json["p"].clone())?,
            dq_dt: serde_json::from_value(json["dq_dt"].clone())?,
            dp_dt: serde_json::from_value(json["dp_dt"].clone())?,
            prices: serde_json::from_value(json["prices"].clone())?,
            timestamps: Vec::new(),
        }
    };

    let n = data.q.len();
    let coord_dim = data.q[0].len();
    println!("Loaded {} samples (coord_dim={})", n, coord_dim);

    // Train/test split
    let train_size = (n as f64 * 0.8) as usize;
    let q_train = &data.q[..train_size];
    let p_train = &data.p[..train_size];
    let dq_train = &data.dq_dt[..train_size];
    let dp_train = &data.dp_dt[..train_size];

    let q_test = &data.q[train_size..];
    let p_test = &data.p[train_size..];
    let dq_test = &data.dq_dt[train_size..];
    let dp_test = &data.dp_dt[train_size..];

    println!("Train: {}, Test: {}", train_size, n - train_size);

    // Create model
    let mut model = HamiltonianNN::new(coord_dim, args.hidden_dim, args.num_layers);
    let n_params = model.num_parameters();
    println!(
        "\nModel: HamiltonianNN (hidden_dim={}, layers={}, params={})",
        args.hidden_dim, args.num_layers, n_params
    );

    // Optimizer
    let mut optimizer = SGDOptimizer::new(n_params, args.lr, args.momentum);

    // Training loop
    println!("\nTraining for {} epochs (batch_size={})...\n", args.epochs, args.batch_size);

    let pb = ProgressBar::new(args.epochs as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("##-"),
    );

    let mut best_val_loss = f64::INFINITY;
    let mut best_params = model.parameters();

    for epoch in 0..args.epochs {
        // Mini-batch training
        let mut epoch_loss = 0.0;
        let mut n_batches = 0;

        let mut indices: Vec<usize> = (0..train_size).collect();
        // Simple shuffle using rand
        for i in (1..indices.len()).rev() {
            let j = rand::random::<usize>() % (i + 1);
            indices.swap(i, j);
        }

        for batch_start in (0..train_size).step_by(args.batch_size) {
            let batch_end = (batch_start + args.batch_size).min(train_size);
            let batch_indices = &indices[batch_start..batch_end];

            let q_batch: Vec<Vec<f64>> = batch_indices.iter().map(|&i| q_train[i].clone()).collect();
            let p_batch: Vec<Vec<f64>> = batch_indices.iter().map(|&i| p_train[i].clone()).collect();
            let dq_batch: Vec<Vec<f64>> = batch_indices.iter().map(|&i| dq_train[i].clone()).collect();
            let dp_batch: Vec<Vec<f64>> = batch_indices.iter().map(|&i| dp_train[i].clone()).collect();

            let (loss, gradients) = compute_loss_and_gradients(
                &model, &q_batch, &p_batch, &dq_batch, &dp_batch,
            );

            let mut params = model.parameters();
            optimizer.step(&mut params, &gradients);
            model.set_parameters(&params);

            epoch_loss += loss;
            n_batches += 1;
        }

        epoch_loss /= n_batches as f64;

        // Validation loss (on full test set, capped for speed)
        let val_size = q_test.len().min(200);
        let (val_loss, _) = compute_loss_and_gradients(
            &model,
            &q_test[..val_size],
            &p_test[..val_size],
            &dq_test[..val_size],
            &dp_test[..val_size],
        );

        if val_loss < best_val_loss {
            best_val_loss = val_loss;
            best_params = model.parameters();
        }

        // Learning rate decay (cosine annealing)
        let progress = epoch as f64 / args.epochs as f64;
        let lr = args.lr * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
        optimizer.learning_rate = lr;

        if epoch % 10 == 0 || epoch == args.epochs - 1 {
            pb.set_message(format!(
                "train={:.6}, val={:.6}, lr={:.6}",
                epoch_loss, val_loss, lr
            ));
        }
        pb.inc(1);
    }
    pb.finish_with_message("Training complete!");

    // Restore best model
    model.set_parameters(&best_params);

    // Final evaluation
    println!("\n{}", "=".repeat(60));
    println!("TRAINING RESULTS");
    println!("{}", "=".repeat(60));
    println!("Best validation loss: {:.8}", best_val_loss);

    // Energy conservation check
    if !q_test.is_empty() {
        let (traj_q, traj_p) = integrate_trajectory(
            &model, &q_test[0], &p_test[0], 0.1, 100,
        );
        let energies = energy_along_trajectory(&model, &traj_q, &traj_p);
        let e_mean: f64 = energies.iter().sum::<f64>() / energies.len() as f64;
        let e_std: f64 = {
            let var: f64 = energies.iter().map(|&e| (e - e_mean).powi(2)).sum::<f64>()
                / energies.len() as f64;
            var.sqrt()
        };
        let e_drift = energies.last().unwrap() - energies.first().unwrap();

        println!("\nEnergy conservation (100-step trajectory):");
        println!("  H mean: {:.6}", e_mean);
        println!("  H std:  {:.6}", e_std);
        println!("  H drift: {:.6}", e_drift);
    }

    // Save model
    let output_dir = std::path::Path::new(&args.output).parent().unwrap();
    std::fs::create_dir_all(output_dir)?;

    let model_json = serde_json::to_string_pretty(&model)?;
    std::fs::write(&args.output, &model_json)?;
    println!("\nModel saved to {}", args.output);

    // Also save in bincode for fast loading
    let bincode_path = args.output.replace(".json", ".bin");
    let model_bytes = bincode::serialize(&model)?;
    std::fs::write(&bincode_path, &model_bytes)?;
    println!("Model (binary) saved to {}", bincode_path);

    Ok(())
}
