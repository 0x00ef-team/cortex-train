//! Cortex-AI: create a model from a preset and train it on code.
//!
//! Usage:
//!   cargo run --release --example train_code -- 8m
//!   cargo run --release --example train_code -- 1b   # prints config only

use cortex_train::{checkpoint, preset, AdamW, Dataset, SGD, Transformer};
use std::collections::HashMap;

const CODE: &str = r#"
fn add(a: i32, b: i32) -> i32 { a + b }
fn main() { let x = add(1, 2); println!("{}", x); }
struct Point { x: f64, y: f64 }
impl Point { fn norm(&self) -> f64 { (self.x*self.x + self.y*self.y).sqrt() } }
fn factorial(n: u64) -> u64 { if n <= 1 { 1 } else { n * factorial(n-1) } }
fn greet(name: &str) -> String { format!("hello {}", name) }
"#;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let name = args.first().map(String::as_str).unwrap_or("8m");
    let p = preset(name);
    println!("{}: dim={} layers={} heads={} hidden={} vocab={}",
        p.name, p.dim, p.layers, p.heads, p.hidden, p.vocab);
    println!("parameters: {:.2}M (f32 weights = {:.1} MiB, with Adam states ≈ {:.1} GiB)",
        p.params() as f64 / 1e6,
        p.params() as f64 * 4.0 / 1048576.0,
        p.params() as f64 * 16.0 / 1073741824.0);

    if name == "300m" || name == "cortex-ai-300m" {
        let mut chars: Vec<char> = CODE.chars().collect();
        chars.sort(); chars.dedup();
        let stoi: HashMap<char, usize> = chars.iter().enumerate().map(|(i, c)| (*c, i)).collect();
        let ids: Vec<usize> = CODE.chars().map(|c| stoi[&c]).collect();
        let ds = Dataset::windows(&ids, 16);
        let model: Transformer = p.build();
        // SGD keeps state to weights + grads (~2.1 GiB) instead of Adam's ~4.3 GiB.
        let params = model.params();
        let mut opt = SGD::new(params, 0.05);
        let steps: usize = std::env::var("CORTEX_STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
        let started = std::time::Instant::now();
        for step in 0..steps {
            let (x, y) = ds.get(step % ds.len());
            opt.zero_grad();
            let loss = model.loss(x, y);
            loss.backward();
            opt.step();
            println!("300m step {step} loss {:.4} elapsed {:.1}s", loss.values()[0], started.elapsed().as_secs_f64());
        }
        return;
    }

    if name == "1b" || name == "cortex-ai-1b" {
        println!("1B model created. Training from scratch requires a large corpus and GPU cluster;");
        println!("this demo trains the 8m preset instead: cargo run --release --example train_code -- 8m");
        return;
    }

    let mut chars: Vec<char> = CODE.chars().collect();
    chars.sort(); chars.dedup();
    let stoi: HashMap<char, usize> = chars.iter().enumerate().map(|(i, c)| (*c, i)).collect();
    let itos: Vec<char> = chars.clone();
    let ids: Vec<usize> = CODE.chars().map(|c| stoi[&c]).collect();
    let ds = Dataset::windows(&ids, 32);
    // The preset's nominal vocab is for scaling estimates; the demo model uses
    // exactly the characters present in the corpus so sampling stays in range.
    let model: Transformer = Transformer::new(chars.len(), p.dim, p.layers, p.heads, p.hidden, p.seed);
    println!("demo vocab (corpus chars): {}", chars.len());
    let params = model.params();
    let mut opt = AdamW::new(params, 0.003).with_weight_decay(0.001);
    let mut order: Vec<usize> = (0..ds.len()).collect();
    let mut rng = 99u64;
    for step in 0..1200 {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        order.sort_by_key(|_| rng);
        let i = order[step % order.len()];
        let (x, y) = ds.get(i);
        opt.zero_grad();
        let loss = model.loss(x, y);
        loss.backward();
        opt.step();
        if step % 200 == 0 { println!("step {step:>4} loss {:.4}", loss.values()[0]); }
    }

    let prompt: Vec<usize> = "fn ".chars().map(|c| stoi[&c]).collect();
    let out = model.generate_sample(&prompt, 160, 0.7, 123);
    println!("\n--- Cortex-AI generated code ---\n{}",
        out.iter().map(|&i| itos[i]).collect::<String>());

    let path = std::env::temp_dir().join("cortex-ai-code.ctrk");
    checkpoint::save(&path, &model.named_params()).unwrap();
    println!("checkpoint: {}", path.display());
}
