//! Test the LOCAL BPE checkpoint on several code prompts. No uploads.

use cortex_train::cuda::Cuda;
use cortex_train::{Bpe, GpuConfig, GpuTransformer};
use std::fs;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let bytes = fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/datasets/code.txt"))?;
    let bpe = Bpe::train(&bytes, 1_000_000, 1024);
    let cfg = GpuConfig {
        vocab: bpe.vocab(), dim: 128, layers: 6, heads: 4, hidden: 512,
        seq: 32, batch: 32, seed: 99, lr: 5e-6,
    };
    let mut model = GpuTransformer::new(Cuda::new()?, cfg)?;
    model.load_checkpoint("/tmp/cortex-bpe-model.ctrk")?;
    model.set_max_temp(75);
    println!("vocab {}, checkpoint loaded (local only)\n", bpe.vocab());

    for prompt in ["def ", "fn ", "class ", "import ", "# ", "x = "] {
        let enc = bpe.encode(prompt.as_bytes());
        let mut ids: Vec<i32> = enc.iter().map(|&v| v as i32).collect();
        // pad to the model's window size with a whitespace token
        ids.resize(32, bpe.encode(b" ").first().copied().unwrap_or(0) as i32);
        let out = model.generate(&ids, 120)?;
        let decoded = bpe.decode(&out.iter().map(|&v| v as u16).collect::<Vec<_>>());
        let text = String::from_utf8_lossy(&decoded);
        println!("--- prompt {prompt:?} ---");
        println!("ids[:40] = {:?}", &out[enc.len()..(enc.len()+40).min(out.len())]);
        println!("raw = {:?}\n", &decoded[..decoded.len().min(200)]);
    }
    Ok(())
}
