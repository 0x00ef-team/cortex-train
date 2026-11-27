//! Multi-layer GPU transformer training (reusable `GpuTransformer` module).

use cortex_train::cuda::Cuda;
use cortex_train::{Corpus, GpuConfig, GpuTransformer};

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let cfg = GpuConfig {
        vocab: 256, dim: 128, layers: 6, heads: 4, hidden: 512,
        seq: 32, batch: 32, seed: 99, lr: 0.00005,
    };
    let n = cfg.batch * cfg.seq;
    let corpus_path = concat!(env!("CARGO_MANIFEST_DIR"), "/datasets/code.txt");
    let corpus = Corpus::load(&[corpus_path])?;
    println!("corpus: {corpus_path}, {} tokens", corpus.len());
    let all = &corpus.tokens;
    let mut rng = 31337u64;
    let window = |start: usize| -> (Vec<i32>, Vec<i32>) {
        let ids = all[start..start + n].iter().map(|&b| b as i32).collect();
        let targets = all[start + 1..start + 1 + n].iter().map(|&b| b as i32).collect();
        (ids, targets)
    };
    let (_ids0, _targets0) = window(0);

    let mut model = GpuTransformer::new(Cuda::new()?, cfg)?;
    model.set_max_temp(75); // never let the GPU reach 79C
    for step in 1..=2000 {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        let start = (rng as usize) % (all.len() - n - 1);
        let (ids, targets) = window(start);
        let loss = model.train_batch(&ids, &targets)?;
        if step % 250 == 0 { println!("step {step:>3} loss {loss:.4}"); }
    }
    let prompt: Vec<i32> = b"def ".iter().map(|&c| c as i32).collect();
    let out = model.generate(&prompt, 240)?;
    println!("\n--- generated ---\n{}", corpus.decode(&out.iter().map(|&i| i as usize).collect::<Vec<_>>()));
    let path = std::env::temp_dir().join("cortex-gpu-transformer.ctrk");
    model.save_checkpoint(&path)?;
    println!("checkpoint: {}", path.display());
    Ok(())
}
