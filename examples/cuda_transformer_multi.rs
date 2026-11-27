//! Multi-layer GPU transformer training (reusable `GpuTransformer` module).

use cortex_train::cuda::Cuda;
use cortex_train::{GpuConfig, GpuTransformer};

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let cfg = GpuConfig {
        vocab: 32, dim: 32, layers: 2, heads: 2, hidden: 64,
        seq: 8, batch: 8, seed: 99, lr: 0.001,
    };
    let n = cfg.batch * cfg.seq;
    let ids: Vec<i32> = (0..n).map(|i| ((i % 13) + 2) as i32).collect();
    let targets: Vec<i32> = (0..n).map(|i| ((i + 1) % 13 + 2) as i32).collect();

    let mut model = GpuTransformer::new(Cuda::new()?, cfg)?;
    for step in 1..=80 {
        let loss = model.train_batch(&ids, &targets)?;
        if step % 10 == 0 { println!("step {step:>3} loss {loss:.4}"); }
    }
    Ok(())
}
