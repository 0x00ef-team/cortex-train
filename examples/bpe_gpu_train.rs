//! Train a GPU transformer on BPE tokens instead of raw bytes.
//!
//! The tokenizer is trained locally from `datasets/code.txt`; nothing is
//! downloaded or uploaded during training.

use cortex_train::cuda::Cuda;
use cortex_train::{Bpe, GpuConfig, GpuTransformer};
use std::time::Instant;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let bytes = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/datasets/code.txt"))?;
    let t0 = Instant::now();
    let bpe = Bpe::train(&bytes, 1_000_000, 1024);
    println!("BPE vocab {}, trained in {:.1}s", bpe.vocab(), t0.elapsed().as_secs_f64());
    let ids = bpe.encode(&bytes);
    let avg_bytes = bytes.len() as f64 / ids.len() as f64;
    println!("encoded {} tokens, avg {:.2} bytes/token, compression {:.2}x",
             ids.len(), avg_bytes, bytes.len() as f64 / ids.len() as f64);

    let cfg = GpuConfig {
        vocab: bpe.vocab(), dim: 128, layers: 6, heads: 4, hidden: 512,
        seq: 32, batch: 32, seed: 99, lr: 5e-5,
    };
    let n = cfg.batch * cfg.seq;
    let mut model = GpuTransformer::new(Cuda::new()?, cfg)?;
    model.set_max_temp(75);
    let mut rng = 0xB0FEu64;
    let mut running = 0f64;
    let started = Instant::now();
    let steps: usize = std::env::var("CORTEX_STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(5000);
    for step in 1..=steps {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        let start = (rng as usize) % (ids.len() - n - 1);
        let x: Vec<i32> = ids[start..start + n].iter().map(|&v| v as i32).collect();
        let y: Vec<i32> = ids[start + 1..start + 1 + n].iter().map(|&v| v as i32).collect();
        running += model.train_batch(&x, &y)? as f64;
        if step % 250 == 0 {
            let loss = running / 250.0;
            running = 0.0;
            let bits = loss / std::f64::consts::LN_2 / avg_bytes;
            println!("step {step:>5}/{steps} loss {loss:.4}  {bits:.3} bits/byte  lr {:?}",
                     std::env::var("CORTEX_LR").ok());
        }
    }
    let prompt = bpe.encode(b"def ");
    let out = model.generate(&prompt.iter().map(|&v| v as i32).collect::<Vec<_>>(), 200)?;
    let decoded = bpe.decode(&out.iter().map(|&v| v as u16).collect::<Vec<_>>());
    let text = String::from_utf8_lossy(&decoded);
    println!("\n--- generated ---\n{text}");
    model.save_checkpoint("/tmp/cortex-bpe-model.ctrk")?;
    println!("saved locally to /tmp/cortex-bpe-model.ctrk in {:.1}m", started.elapsed().as_secs_f64()/60.0);
    Ok(())
}
