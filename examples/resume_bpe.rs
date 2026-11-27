//! Continue BPE GPU training locally, without uploading anything.
//!
//!   ./target/release/examples/resume_bpe /tmp/cortex-bpe-model.ctrk 50000

use cortex_train::cuda::Cuda;
use cortex_train::{Bpe, GpuConfig, GpuTransformer};
use std::path::PathBuf;
use std::time::Instant;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = PathBuf::from(args.first().cloned().unwrap_or_else(|| "/tmp/cortex-bpe-model.ctrk".into()));
    let steps: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(50000);
    let lr: f32 = std::env::var("CORTEX_LR").ok().and_then(|v| v.parse().ok()).unwrap_or(5e-6);

    let bytes = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/datasets/code.txt"))?;
    let bpe = Bpe::train(&bytes, 1_000_000, 1024);
    let ids = bpe.encode(&bytes);
    let avg_bytes = bytes.len() as f64 / ids.len() as f64;
    println!("BPE vocab {}, {} tokens, {:.2} bytes/token", bpe.vocab(), ids.len(), avg_bytes);

    let cfg = GpuConfig {
        vocab: bpe.vocab(), dim: 128, layers: 6, heads: 4, hidden: 512,
        seq: 32, batch: 32, seed: 99, lr,
    };
    let n = cfg.batch * cfg.seq;
    let mut model = GpuTransformer::new(Cuda::new()?, cfg)?;
    model.load_checkpoint(&path)?;
    model.set_max_temp(75);
    println!("resumed {}, lr {lr:.1e}, {steps} steps", path.display());

    let best_path = path.with_extension("best.ctrk");
    let mut best_loss = f64::INFINITY;
    let mut bad = 0usize;
    let mut rng = 0xB0FEu64;
    let mut running = 0f64;
    let started = Instant::now();
    for step in 1..=steps {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        let start = (rng as usize) % (ids.len() - n - 1);
        let x: Vec<i32> = ids[start..start + n].iter().map(|&v| v as i32).collect();
        let y: Vec<i32> = ids[start + 1..start + 1 + n].iter().map(|&v| v as i32).collect();
        running += model.train_batch(&x, &y)? as f64;
        if step % 250 == 0 {
            let loss = running / 250.0;
            running = 0.0;
            if loss < best_loss {
                best_loss = loss;
                model.save_checkpoint(&best_path)?;
                bad = 0;
            } else if loss > best_loss * 1.1 {
                bad += 1;
                if bad >= 2 {
                    eprintln!("[rollback] loss {loss:.3} > best {best_loss:.3}");
                    model.load_checkpoint(&best_path)?;
                    model.reset_optimizer()?;
                    model.set_lr(lr * 0.5);
                    bad = 0;
                }
            } else { bad = 0; }

            let bits = loss / std::f64::consts::LN_2 / avg_bytes;
            let elapsed = started.elapsed().as_secs_f64();
            let eta = elapsed / step as f64 * (steps - step) as f64;
            let pct = 100.0 * step as f64 / steps as f64;
            let temp = cortex_train::thermal::gpu_temp_c()
                .map(|t| format!("{t}C")).unwrap_or_else(|| "n/a".into());
            println!("step {step:>6}/{steps} ({pct:5.1}%) loss {loss:.4} {bits:.3} bits/byte best {best_loss:.4} elapsed {:.1}m eta {:.1}m gpu {temp}",
                     elapsed / 60.0, eta / 60.0);
            let status = format!(
                "{{\"step\":{},\"total\":{},\"loss\":{},\"bits_per_byte\":{},\"elapsed_s\":{},\"eta_s\":{},\"temp_c\":{}}}",
                step, steps, loss, bits, elapsed as u64, eta as u64,
                cortex_train::thermal::gpu_temp_c().unwrap_or(0));
            let _ = std::fs::write("/tmp/cortex-train-status.json", status);
        }
        if step % 5000 == 0 { model.save_checkpoint(&path)?; }
    }
    let prompt = bpe.encode(b"def ");
    let out = model.generate(&prompt.iter().map(|&v| v as i32).collect::<Vec<_>>(), 240)?;
    let decoded = bpe.decode(&out.iter().map(|&v| v as u16).collect::<Vec<_>>());
    println!("\n--- generated ---\n{}", String::from_utf8_lossy(&decoded));
    model.save_checkpoint(&path)?;
    println!("saved locally to {} in {:.1}m", path.display(), started.elapsed().as_secs_f64()/60.0);
    Ok(())
}
