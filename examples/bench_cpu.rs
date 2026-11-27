use cortex_train::{AdamW, Corpus, Dataset, Transformer};
use std::time::Instant;

fn main() -> std::io::Result<()> {
    let corpus = Corpus::load(&[concat!(env!("CARGO_MANIFEST_DIR"), "/datasets/code.txt")])?;
    let ids: Vec<usize> = corpus.tokens[..4096].to_vec();
    let ds = Dataset::windows(&ids, 32);
    let model = Transformer::new(256, 128, 6, 4, 512, 99);
    let params = model.params();
    let mut opt = AdamW::new(params, 5e-5);
    let t = Instant::now();
    let steps = 20;
    for s in 0..steps {
        let (x, y) = ds.get(s % ds.len());
        opt.zero_grad();
        model.loss(x, y).backward();
        opt.step();
    }
    let secs = t.elapsed().as_secs_f64();
    println!("CPU: {steps} steps, {:.3}s/step, {} tok/s (seq 32, batch 1)",
             secs / steps as f64, (steps * 32) as f64 / secs);
    Ok(())
}
