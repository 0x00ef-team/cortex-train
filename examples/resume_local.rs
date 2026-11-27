//! Continue training the LOCAL checkpoint. Nothing is committed or uploaded.
//!
//!   cargo run --release --example resume_local -- /tmp/cortex-gpu-transformer.ctrk 3000

use cortex_train::cuda::Cuda;
use cortex_train::{Corpus, GpuConfig, GpuTransformer};
use std::path::PathBuf;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = PathBuf::from(args.first().cloned().unwrap_or_else(|| "/tmp/cortex-gpu-transformer.ctrk".into()));
    let steps: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(3000);

    let lr: f32 = std::env::var("CORTEX_LR").ok().and_then(|v| v.parse().ok()).unwrap_or(1e-5);
    let cfg = GpuConfig {
        vocab: 256, dim: 128, layers: 6, heads: 4, hidden: 512,
        seq: 32, batch: 32, seed: 99, lr,
    };
    let n = cfg.batch * cfg.seq;
    let corpus = Corpus::load(&[concat!(env!("CARGO_MANIFEST_DIR"), "/datasets/code.txt")])?;
    let all = &corpus.tokens;

    let mut model = GpuTransformer::new(Cuda::new()?, cfg)?;
    model.set_max_temp(75); // never let the GPU reach 79C
    model.load_checkpoint(&path)?;
    println!("resumed from {} ({} tokens corpus)", path.display(), all.len());

    let mut rng = 0xC0FFEEu64;
    let mut running = 0f32;
    for step in 1..=steps {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        let start = (rng as usize) % (all.len() - n - 1);
        let ids: Vec<i32> = all[start..start + n].iter().map(|&b| b as i32).collect();
        let targets: Vec<i32> = all[start + 1..start + 1 + n].iter().map(|&b| b as i32).collect();
        let loss = model.train_batch(&ids, &targets)?;
        running += loss;
        if step % 50 == 0 {
            println!("step {step:>5} avg_loss {:.4}", running / 250.0);
            running = 0.0;
        }
    }

    let prompt: Vec<i32> = b"def ".iter().map(|&c| c as i32).collect();
    let out = model.generate(&prompt, 240)?;
    println!("\n--- generated (local model only) ---\n{}",
        corpus.decode(&out.iter().map(|&i| i as usize).collect::<Vec<_>>()));
    model.save_checkpoint(&path)?;
    println!("saved back to {} (local only)", path.display());
    Ok(())
}
