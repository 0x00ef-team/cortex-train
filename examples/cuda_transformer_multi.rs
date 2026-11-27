//! Multi-layer GPU transformer training (reusable `GpuTransformer` module).

use cortex_train::cuda::Cuda;
use cortex_train::{Corpus, GpuConfig, GpuTransformer};

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let cfg = GpuConfig {
        vocab: 256, dim: 32, layers: 2, heads: 2, hidden: 64,
        seq: 16, batch: 8, seed: 99, lr: 0.001,
    };
    let n = cfg.batch * cfg.seq;
    let corpus_path = concat!(env!("CARGO_MANIFEST_DIR"), "/datasets/code.txt");
    let corpus = Corpus::load(&[corpus_path])?;
    println!("corpus: {corpus_path}, {} tokens", corpus.len());
    let all = &corpus.tokens;
    let start = 0usize;
    let ids: Vec<i32> = all[start..start + n].iter().map(|&b| b as i32).collect();
    let targets: Vec<i32> = all[start + 1..start + 1 + n].iter().map(|&b| b as i32).collect();

    let mut model = GpuTransformer::new(Cuda::new()?, cfg)?;
    for step in 1..=80 {
        let loss = model.train_batch(&ids, &targets)?;
        if step % 10 == 0 { println!("step {step:>3} loss {loss:.4}"); }
    }
    Ok(())
}
