//! Run predictions using a trained HNN model.
//!
//! Usage:
//!   cargo run --release --bin predict -- --model output/model.json --horizon 20
//!   cargo run --release --bin predict -- --model output/model.bin --horizon 50

use anyhow::Result;
use clap::Parser;
use hamiltonian_nn_trading::nn::HamiltonianNN;
use hamiltonian_nn_trading::integrator::integrate_trajectory;
use hamiltonian_nn_trading::utils::energy_along_trajectory;
use hamiltonian_nn_trading::data::PhaseSpaceData;

#[derive(Parser, Debug)]
#[command(name = "predict", about = "Run HNN predictions")]
struct Args {
    /// Path to saved model (JSON or bincode)
    #[arg(long, default_value = "output/model.json")]
    model: String,

    /// Prediction horizon (number of steps)
    #[arg(long, default_value_t = 20)]
    horizon: usize,

    /// Integration time step
    #[arg(long, default_value_t = 0.1)]
    dt: f64,

    /// Path to phase space data
    #[arg(long, default_value = "data/BTCUSDT_phase_space.json")]
    data: String,

    /// Starting index in the data
    #[arg(long, default_value_t = 0)]
    start_idx: usize,

    /// Number of predictions to make
    #[arg(long, default_value_t = 10)]
    n_predictions: usize,

    /// Output CSV path
    #[arg(long, default_value = "output/predictions.csv")]
    output: String,

    /// Compare integration methods
    #[arg(long)]
    compare_methods: bool,
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    // Load model
    println!("Loading model from {}...", args.model);
    let model: HamiltonianNN = if args.model.ends_with(".bin") {
        let bytes = std::fs::read(&args.model)?;
        bincode::deserialize(&bytes)?
    } else {
        let json_str = std::fs::read_to_string(&args.model)?;
        serde_json::from_str(&json_str)?
    };

    println!(
        "Model loaded: coord_dim={}, params={}",
        model.coord_dim,
        model.num_parameters()
    );

    // Load data
    println!("Loading data from {}...", args.data);
    let json_str = std::fs::read_to_string(&args.data)?;
    let json: serde_json::Value = serde_json::from_str(&json_str)?;

    let data = PhaseSpaceData {
        q: serde_json::from_value(json["q"].clone())?,
        p: serde_json::from_value(json["p"].clone())?,
        dq_dt: serde_json::from_value(json["dq_dt"].clone())?,
        dp_dt: serde_json::from_value(json["dp_dt"].clone())?,
        prices: serde_json::from_value(json["prices"].clone())?,
        timestamps: Vec::new(),
    };

    println!("Data loaded: {} samples", data.q.len());

    // Create output directory
    let output_dir = std::path::Path::new(&args.output).parent().unwrap();
    std::fs::create_dir_all(output_dir)?;

    // Run predictions
    println!(
        "\nRunning {} predictions (horizon={}, dt={})...\n",
        args.n_predictions, args.horizon, args.dt
    );

    let mut all_predictions = Vec::new();

    let end_idx = (args.start_idx + args.n_predictions).min(data.q.len());

    for i in args.start_idx..end_idx {
        let q0 = &data.q[i];
        let p0 = &data.p[i];

        // Integrate trajectory
        let (traj_q, traj_p) = integrate_trajectory(
            &model, q0, p0, args.dt, args.horizon,
        );

        // Compute energy along trajectory
        let energies = energy_along_trajectory(&model, &traj_q, &traj_p);

        // Predicted change
        let dq_predicted = traj_q.last().unwrap()[0] - traj_q[0][0];
        let dp_predicted = traj_p.last().unwrap()[0] - traj_p[0][0];

        let energy_drift = energies.last().unwrap() - energies.first().unwrap();
        let energy_std = {
            let mean: f64 = energies.iter().sum::<f64>() / energies.len() as f64;
            let var: f64 = energies.iter().map(|&e| (e - mean).powi(2)).sum::<f64>()
                / energies.len() as f64;
            var.sqrt()
        };

        // Signal generation
        let signal = if dq_predicted.abs() > 0.5 {
            if dq_predicted > 0.0 { "BUY" } else { "SELL" }
        } else {
            "HOLD"
        };

        println!(
            "Step {:4}: q0={:+.4}, p0={:+.4} -> dq={:+.6}, dp={:+.6}, E_drift={:+.6}, signal={}",
            i, q0[0], p0[0], dq_predicted, dp_predicted, energy_drift, signal
        );

        all_predictions.push(vec![
            i as f64,
            q0[0],
            p0[0],
            dq_predicted,
            dp_predicted,
            energies[0],
            energy_drift,
            energy_std,
            data.prices[i],
        ]);
    }

    // Compare integration methods
    if args.compare_methods && !data.q.is_empty() {
        println!("\n{}", "=".repeat(60));
        println!("INTEGRATION METHOD COMPARISON");
        println!("{}", "=".repeat(60));

        let q0 = &data.q[args.start_idx];
        let p0 = &data.p[args.start_idx];

        // Leapfrog (symplectic)
        let (traj_lf_q, traj_lf_p) = integrate_trajectory(
            &model, q0, p0, args.dt, args.horizon,
        );
        let e_lf = energy_along_trajectory(&model, &traj_lf_q, &traj_lf_p);
        let drift_lf = e_lf.last().unwrap() - e_lf.first().unwrap();

        // Euler (non-symplectic)
        let mut q_euler = q0.clone();
        let mut p_euler = p0.clone();
        let mut e_euler_start = model.hamiltonian(&q_euler, &p_euler);
        for _ in 0..args.horizon {
            let (dq, dp) = model.time_derivative(&q_euler, &p_euler);
            for d in 0..q_euler.len() {
                q_euler[d] += args.dt * dq[d];
                p_euler[d] += args.dt * dp[d];
            }
        }
        let e_euler_end = model.hamiltonian(&q_euler, &p_euler);
        let drift_euler = e_euler_end - e_euler_start;

        println!(
            "\n{:<15} {:>15} {:>15}",
            "Method", "Energy Drift", "Final q"
        );
        println!("{}", "-".repeat(47));
        println!(
            "{:<15} {:>15.8} {:>15.8}",
            "Leapfrog",
            drift_lf,
            traj_lf_q.last().unwrap()[0]
        );
        println!(
            "{:<15} {:>15.8} {:>15.8}",
            "Euler",
            drift_euler,
            q_euler[0]
        );
        println!(
            "\nLeapfrog energy drift is typically 10-100x smaller than Euler."
        );
    }

    // Save predictions to CSV
    let mut wtr = csv::Writer::from_path(&args.output)?;
    wtr.write_record([
        "step", "q0", "p0", "dq_pred", "dp_pred",
        "energy", "energy_drift", "energy_std", "price",
    ])?;
    for row in &all_predictions {
        let record: Vec<String> = row.iter().map(|v| format!("{:.8}", v)).collect();
        wtr.write_record(&record)?;
    }
    wtr.flush()?;
    println!("\nPredictions saved to {}", args.output);

    Ok(())
}
