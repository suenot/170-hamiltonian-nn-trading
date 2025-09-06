//! Fetch market data from Bybit and construct phase space.
//!
//! Usage:
//!   cargo run --bin fetch_data -- --symbol BTCUSDT --interval 5 --limit 5000

use anyhow::Result;
use clap::Parser;
use hamiltonian_nn_trading::data::{
    BybitClient, construct_phase_space, normalize_phase_space,
};

#[derive(Parser, Debug)]
#[command(name = "fetch_data", about = "Fetch Bybit data for HNN trading")]
struct Args {
    /// Trading symbol (e.g., BTCUSDT)
    #[arg(long, default_value = "BTCUSDT")]
    symbol: String,

    /// Candle interval in minutes
    #[arg(long, default_value = "5")]
    interval: String,

    /// Number of candles to fetch
    #[arg(long, default_value_t = 5000)]
    limit: usize,

    /// Moving average window for phase space
    #[arg(long, default_value_t = 20)]
    ma_window: usize,

    /// Output directory
    #[arg(long, default_value = "data")]
    output: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    println!("Fetching data from Bybit: {}", args.symbol);
    println!("  Interval: {} min", args.interval);
    println!("  Limit: {} candles", args.limit);

    let client = BybitClient::new();
    let candles = client
        .fetch_extended(&args.symbol, &args.interval, args.limit)
        .await?;

    println!("Fetched {} candles", candles.len());

    if candles.len() < 2 {
        anyhow::bail!("Not enough data fetched");
    }

    println!(
        "Date range: {} to {}",
        candles.first().unwrap().timestamp,
        candles.last().unwrap().timestamp
    );
    println!(
        "Price range: {:.2} to {:.2}",
        candles.iter().map(|c| c.close).fold(f64::INFINITY, f64::min),
        candles.iter().map(|c| c.close).fold(f64::NEG_INFINITY, f64::max),
    );

    // Construct phase space
    println!("\nConstructing phase space (MA window: {})...", args.ma_window);
    let phase_data = construct_phase_space(&candles, args.ma_window);
    println!("Phase space samples: {}", phase_data.q.len());

    if phase_data.q.is_empty() {
        anyhow::bail!("No valid phase space data constructed");
    }

    // Show statistics
    let q_vals: Vec<f64> = phase_data.q.iter().map(|v| v[0]).collect();
    let p_vals: Vec<f64> = phase_data.p.iter().map(|v| v[0]).collect();

    println!(
        "q range: [{:.6}, {:.6}]",
        q_vals.iter().cloned().fold(f64::INFINITY, f64::min),
        q_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    );
    println!(
        "p range: [{:.6}, {:.6}]",
        p_vals.iter().cloned().fold(f64::INFINITY, f64::min),
        p_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    );

    // Normalize
    let (normalized, stats) = normalize_phase_space(&phase_data);
    println!("\nNormalization stats:");
    println!("  q_mean: {:?}", stats.q_mean);
    println!("  q_std:  {:?}", stats.q_std);
    println!("  p_mean: {:?}", stats.p_mean);
    println!("  p_std:  {:?}", stats.p_std);

    // Save to files
    std::fs::create_dir_all(&args.output)?;

    // Save raw candles as CSV
    let candle_path = format!("{}/{}_candles.csv", args.output, args.symbol);
    let mut wtr = csv::Writer::from_path(&candle_path)?;
    wtr.write_record(["timestamp", "open", "high", "low", "close", "volume"])?;
    for c in &candles {
        wtr.write_record(&[
            c.timestamp.to_string(),
            format!("{:.4}", c.open),
            format!("{:.4}", c.high),
            format!("{:.4}", c.low),
            format!("{:.4}", c.close),
            format!("{:.4}", c.volume),
        ])?;
    }
    wtr.flush()?;
    println!("\nSaved candles to {}", candle_path);

    // Save phase space as JSON
    let phase_path = format!("{}/{}_phase_space.json", args.output, args.symbol);
    let json_data = serde_json::json!({
        "q": normalized.q,
        "p": normalized.p,
        "dq_dt": normalized.dq_dt,
        "dp_dt": normalized.dp_dt,
        "prices": normalized.prices,
        "stats": {
            "q_mean": stats.q_mean,
            "q_std": stats.q_std,
            "p_mean": stats.p_mean,
            "p_std": stats.p_std,
        }
    });
    std::fs::write(&phase_path, serde_json::to_string_pretty(&json_data)?)?;
    println!("Saved phase space to {}", phase_path);

    // Save phase space as CSV for easy inspection
    let csv_path = format!("{}/{}_phase_space.csv", args.output, args.symbol);
    let mut wtr = csv::Writer::from_path(&csv_path)?;
    wtr.write_record(["q", "p", "dq_dt", "dp_dt", "price"])?;
    for i in 0..normalized.q.len() {
        wtr.write_record(&[
            format!("{:.8}", normalized.q[i][0]),
            format!("{:.8}", normalized.p[i][0]),
            format!("{:.8}", normalized.dq_dt[i][0]),
            format!("{:.8}", normalized.dp_dt[i][0]),
            format!("{:.4}", normalized.prices[i]),
        ])?;
    }
    wtr.flush()?;
    println!("Saved phase space CSV to {}", csv_path);

    let train_size = (normalized.q.len() as f64 * 0.8) as usize;
    println!("\nTrain/Test split:");
    println!("  Train: {} samples", train_size);
    println!("  Test:  {} samples", normalized.q.len() - train_size);

    println!("\nData fetching complete!");
    Ok(())
}
