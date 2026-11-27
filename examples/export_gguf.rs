//! Export a CPU-trained model to GGUF so `cortex` can load it.

use cortex_train::{export_gguf, ExportOptions, Transformer};

fn main() -> std::io::Result<()> {
    let vocab = 32usize;
    let model = Transformer::new(vocab, 16, 2, 4, 32, 7);
    let tokens: Vec<Vec<u8>> = (0..vocab).map(|i| {
        if i < 3 { format!("<t{i}>").into_bytes() } else { vec![b'a' + ((i - 3) % 26) as u8] }
    }).collect();
    let path = "/tmp/cortex-train-export.gguf";
    export_gguf(path, &model, ExportOptions {
        tokens: &tokens, context: 128, name: "cortex-train-export",
    })?;
    println!("wrote {path}");
    Ok(())
}
