//! Train a decoder-only transformer from scratch on character-level text.
//!
//! Nothing is hard-coded to a particular model: `Transformer::new(vocab, dim,
//! layers, heads, hidden, seed)` creates any architecture, and the training
//! loop only uses generic `loss.backward()` + `optimizer.step()`.

use cortex_train::{checkpoint, AdamW, Transformer};
use std::collections::HashMap;

const TEXT: &str = "the quick brown fox jumps over the lazy dog. \
the quick brown fox jumps over the lazy dog. \
a journey of a thousand miles begins with a single step. \
to be or not to be that is the question. ";

fn main() {
    let mut chars: Vec<char> = TEXT.chars().collect();
    chars.sort(); chars.dedup();
    let stoi: HashMap<char, usize> = chars.iter().enumerate().map(|(i, c)| (*c, i)).collect();
    let itos: Vec<char> = chars.clone();
    let ids: Vec<usize> = TEXT.chars().map(|c| stoi[&c]).collect();
    let vocab = chars.len();

    let dim = 48usize;
    let block = 12usize;
    let layers = 3usize;
    let heads = 3usize;
    let hidden = 128usize;
    let model = Transformer::new(vocab, dim, layers, heads, hidden, 42);
    let params = model.params();
    let mut opt = AdamW::new(params.clone(), 0.005).with_weight_decay(0.001);

    let mut rng = 12345u64;
    let mut next = || { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; rng };
    let samples = 1500usize;
    for step in 0..samples {
        let start = (next() as usize) % (ids.len() - block - 1);
        let x = &ids[start..start + block];
        let y = &ids[start + 1..start + block + 1];
        opt.zero_grad();
        let loss = model.loss(x, y);
        loss.backward();
        opt.step();
        if step % 150 == 0 || step + 1 == samples {
            println!("step {step:>4}  loss {:.4}", loss.values()[0]);
        }
    }

    let prompt = vec![stoi[&'t'], stoi[&'h'], stoi[&'e']];
    let out = model.generate_sample(&prompt, 160, 0.8, 777);
    let text: String = out.iter().map(|&i| itos[i]).collect();
    println!("\n--- generated ---\n{text}");

    let path = std::env::temp_dir().join("cortex-train-demo.ctrk");
    checkpoint::save(&path, &model.named_params()).expect("save checkpoint");
    let restored = checkpoint::load(&path).expect("load checkpoint");
    println!("checkpoint: {} tensors at {}", restored.len(), path.display());

    // Sanity: the model must produce logits of the expected shape.
    let logits = model.forward(&prompt);
    assert_eq!(logits.shape(), vec![prompt.len(), vocab]);
}
