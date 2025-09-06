//! Phase portrait visualization example.
//!
//! Demonstrates:
//! 1. Creating an HNN
//! 2. Integrating trajectories
//! 3. Computing energy along trajectories
//! 4. Exporting data for plotting
//!
//! Usage:
//!   cargo run --example phase_portrait

use hamiltonian_nn_trading::nn::HamiltonianNN;
use hamiltonian_nn_trading::integrator::{integrate_trajectory, euler_step};
use hamiltonian_nn_trading::utils::energy_along_trajectory;

fn main() {
    println!("=== Hamiltonian Neural Network: Phase Portrait Example ===\n");

    // Create a simple HNN
    let model = HamiltonianNN::new(1, 32, 2);
    println!(
        "Created HNN: coord_dim={}, params={}",
        model.coord_dim,
        model.num_parameters()
    );

    // Initial conditions: multiple starting points
    let initial_conditions: Vec<(f64, f64)> = vec![
        (1.0, 0.0),
        (0.5, 0.5),
        (0.0, 1.0),
        (-0.5, 0.5),
        (1.5, 0.0),
    ];

    let dt = 0.02;
    let n_steps = 500;

    println!("\nIntegrating {} trajectories (dt={}, steps={})...\n",
        initial_conditions.len(), dt, n_steps);

    // Integrate each trajectory and collect data
    let mut all_data: Vec<Vec<f64>> = Vec::new();

    for (idx, &(q0, p0)) in initial_conditions.iter().enumerate() {
        let q_init = vec![q0];
        let p_init = vec![p0];

        // Leapfrog integration
        let (traj_q, traj_p) = integrate_trajectory(
            &model, &q_init, &p_init, dt, n_steps,
        );

        // Energy along trajectory
        let energies = energy_along_trajectory(&model, &traj_q, &traj_p);

        // Statistics
        let e_mean: f64 = energies.iter().sum::<f64>() / energies.len() as f64;
        let e_std: f64 = {
            let var: f64 = energies.iter().map(|&e| (e - e_mean).powi(2)).sum::<f64>()
                / energies.len() as f64;
            var.sqrt()
        };
        let e_drift = energies.last().unwrap() - energies.first().unwrap();

        println!(
            "Trajectory {}: ({:.2}, {:.2}) -> E_mean={:.4}, E_std={:.4}, E_drift={:.6}",
            idx, q0, p0, e_mean, e_std, e_drift
        );

        // Collect data for CSV export
        for i in 0..traj_q.len() {
            all_data.push(vec![
                idx as f64,
                i as f64 * dt,
                traj_q[i][0],
                traj_p[i][0],
                energies[i],
            ]);
        }
    }

    // Compare Leapfrog vs Euler
    println!("\n=== Integration Method Comparison ===\n");
    let q0 = vec![1.0];
    let p0 = vec![0.0];

    // Leapfrog
    let (traj_lf_q, traj_lf_p) = integrate_trajectory(&model, &q0, &p0, dt, n_steps);
    let e_lf = energy_along_trajectory(&model, &traj_lf_q, &traj_lf_p);
    let lf_drift = (e_lf.last().unwrap() - e_lf.first().unwrap()).abs();

    // Euler
    let mut q_euler = q0.clone();
    let mut p_euler = p0.clone();
    let e_euler_start = model.hamiltonian(&q_euler, &p_euler);
    for _ in 0..n_steps {
        let (q_new, p_new) = euler_step(&model, &q_euler, &p_euler, dt);
        q_euler = q_new;
        p_euler = p_new;
    }
    let e_euler_end = model.hamiltonian(&q_euler, &p_euler);
    let euler_drift = (e_euler_end - e_euler_start).abs();

    println!("{:<15} {:>18} {:>18}", "Method", "|Energy Drift|", "Final (q, p)");
    println!("{}", "-".repeat(53));
    println!(
        "{:<15} {:>18.10} ({:>6.3}, {:>6.3})",
        "Leapfrog", lf_drift,
        traj_lf_q.last().unwrap()[0],
        traj_lf_p.last().unwrap()[0]
    );
    println!(
        "{:<15} {:>18.10} ({:>6.3}, {:>6.3})",
        "Euler", euler_drift, q_euler[0], p_euler[0]
    );

    let ratio = if lf_drift > 1e-15 { euler_drift / lf_drift } else { f64::INFINITY };
    println!(
        "\nEuler drift is {:.1}x larger than Leapfrog",
        ratio
    );

    // Export data for external plotting
    let csv_path = "output/phase_portrait_data.csv";
    std::fs::create_dir_all("output").ok();
    match hamiltonian_nn_trading::utils::export_csv(
        csv_path,
        &["trajectory", "time", "q", "p", "energy"],
        &all_data,
    ) {
        Ok(_) => println!("\nPhase portrait data exported to {}", csv_path),
        Err(e) => println!("\nFailed to export CSV: {}", e),
    }

    println!("\n=== Example Complete ===");
    println!("\nTo visualize the phase portrait, plot q vs p from the CSV.");
    println!("Each trajectory index represents a different initial condition.");
    println!("Energy should remain approximately constant along each trajectory");
    println!("(better conservation with Leapfrog than Euler).");
}
