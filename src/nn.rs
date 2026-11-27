//! Composable neural-network modules built on `Tensor`.

use crate::tensor::{cross_entropy, Tensor};

#[derive(Clone)]
pub struct Linear {
    pub weight: Tensor,      // [in, out]
    pub bias: Option<Tensor>, // [out]
}

impl Linear {
    pub fn new(input: usize, output: usize, seed: u64) -> Self {
        let scale = (2.0 / input as f32).sqrt();
        Self {
            weight: Tensor::parameter(Tensor::randn(&[input, output], scale, seed).values(), &[input, output]),
            bias: Some(Tensor::parameter(vec![0.0; output], &[output])),
        }
    }
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let y = x.matmul(&self.weight);
        match &self.bias { Some(b) => y.add(b), None => y }
    }
    pub fn params(&self) -> Vec<Tensor> {
        let mut p = vec![self.weight.clone()];
        if let Some(b) = &self.bias { p.push(b.clone()); }
        p
    }
}

#[derive(Clone)]
pub struct Embedding {
    pub weight: Tensor, // [vocab, dim]
}

impl Embedding {
    pub fn new(vocab: usize, dim: usize, seed: u64) -> Self {
        Self { weight: Tensor::parameter(Tensor::randn(&[vocab, dim], 0.02, seed).values(), &[vocab, dim]) }
    }
    pub fn forward(&self, ids: &[usize]) -> Tensor { Tensor::embedding(&self.weight, ids) }
    pub fn params(&self) -> Vec<Tensor> { vec![self.weight.clone()] }
}

#[derive(Clone)]
pub struct RMSNorm {
    pub weight: Tensor,
    eps: f32,
}

impl RMSNorm {
    pub fn new(dim: usize, eps: f32) -> Self {
        Self { weight: Tensor::parameter(vec![1.0; dim], &[dim]), eps }
    }
    pub fn forward(&self, x: &Tensor) -> Tensor { x.rms_norm(&self.weight, self.eps) }
    pub fn params(&self) -> Vec<Tensor> { vec![self.weight.clone()] }
}

/// One causal self-attention head group with learned Q/K/V/O projections.
#[derive(Clone)]
pub struct CausalAttention {
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    heads: usize,
    dim: usize,
}

impl CausalAttention {
    pub fn heads(&self) -> usize { self.heads }
    pub fn dim(&self) -> usize { self.dim }

    pub fn new(dim: usize, heads: usize, seed: u64) -> Self {
        assert_eq!(dim % heads, 0);
        Self {
            wq: Linear::new(dim, dim, seed + 1),
            wk: Linear::new(dim, dim, seed + 2),
            wv: Linear::new(dim, dim, seed + 3),
            wo: Linear::new(dim, dim, seed + 4),
            heads,
            dim,
        }
    }
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let (sh, hd) = (x.shape(), self.dim / self.heads);
        let t = sh[0];
        let q = self.wq.forward(x).reshape(&[t, self.heads, hd]).transpose01();
        let k = self.wk.forward(x).reshape(&[t, self.heads, hd]).transpose01();
        let v = self.wv.forward(x).reshape(&[t, self.heads, hd]).transpose01();
        let scale = (hd as f32).sqrt().recip();
        let scores = q.matmul(&k.transpose12()).scale(scale);
        // Causal mask: `-inf` above the diagonal.
        let mut mask = vec![0.0f32; self.heads * t * t];
        for h in 0..self.heads { for i in 0..t { for j in (i + 1)..t {
            mask[h*t*t + i*t + j] = -1e9;
        }}}
        let mask = Tensor::new(mask, &[self.heads, t, t]);
        let att = scores.add(&mask).softmax();
        let out = att.matmul(&v).transpose01().reshape(&[t, self.dim]);
        self.wo.forward(&out)
    }
    pub fn params(&self) -> Vec<Tensor> {
        let mut p = Vec::new();
        p.extend(self.wq.params()); p.extend(self.wk.params());
        p.extend(self.wv.params()); p.extend(self.wo.params());
        p
    }
}

#[derive(Clone)]
pub struct TransformerBlock {
    pub attn_norm: RMSNorm,
    pub attn: CausalAttention,
    pub ffn_norm: RMSNorm,
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
}

impl TransformerBlock {
    pub fn new(dim: usize, heads: usize, hidden: usize, seed: u64) -> Self {
        Self {
            attn_norm: RMSNorm::new(dim, 1e-5),
            attn: CausalAttention::new(dim, heads, seed),
            ffn_norm: RMSNorm::new(dim, 1e-5),
            gate: Linear::new(dim, hidden, seed + 10),
            up: Linear::new(dim, hidden, seed + 20),
            down: Linear::new(hidden, dim, seed + 30),
        }
    }
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let a = self.attn.forward(&self.attn_norm.forward(x));
        let h = x.add(&a);
        let g = self.gate.forward(&self.ffn_norm.forward(&h));
        let u = self.up.forward(&self.ffn_norm.forward(&h));
        let f = self.down.forward(&g.silu().mul(&u));
        h.add(&f)
    }
    pub fn params(&self) -> Vec<Tensor> {
        let mut p = Vec::new();
        p.extend(self.attn_norm.params());
        p.extend(self.attn.params());
        p.extend(self.ffn_norm.params());
        p.extend(self.gate.params());
        p.extend(self.up.params());
        p.extend(self.down.params());
        p
    }
}

/// A complete decoder-only transformer you can instantiate with any config.
#[derive(Clone)]
pub struct Transformer {
    pub embed: Embedding,
    pub pos: Embedding,
    pub blocks: Vec<TransformerBlock>,
    pub final_norm: RMSNorm,
    pub head: Linear,
}

impl Transformer {
    pub fn dim(&self) -> usize { self.embed.weight.shape()[1] }
    pub fn vocab(&self) -> usize { self.embed.weight.shape()[0] }
    pub fn layers(&self) -> usize { self.blocks.len() }
    pub fn heads(&self) -> usize { self.blocks.first().map(|b| b.attn.heads()).unwrap_or(1) }
    pub fn hidden(&self) -> usize { self.blocks.first().map(|b| b.gate.weight.shape()[1]).unwrap_or(0) }

    pub fn new(vocab: usize, dim: usize, layers: usize, heads: usize, hidden: usize, seed: u64) -> Self {
        let mut blocks = Vec::with_capacity(layers);
        for i in 0..layers { blocks.push(TransformerBlock::new(dim, heads, hidden, seed + 100 * i as u64)); }
        Self {
            embed: Embedding::new(vocab, dim, seed + 1),
            pos: Embedding::new(1024, dim, seed + 2),
            blocks,
            final_norm: RMSNorm::new(dim, 1e-5),
            head: Linear::new(dim, vocab, seed + 3),
        }
    }

    pub fn forward(&self, ids: &[usize]) -> Tensor {
        let positions: Vec<usize> = (0..ids.len()).collect();
        let mut x = self.embed.forward(ids).add(&self.pos.forward(&positions));
        for b in &self.blocks { x = b.forward(&x); }
        self.head.forward(&self.final_norm.forward(&x))
    }

    pub fn loss(&self, ids: &[usize], targets: &[usize]) -> Tensor {
        cross_entropy(&self.forward(ids), targets)
    }

    pub fn params(&self) -> Vec<Tensor> {
        let mut p = Vec::new();
        p.extend(self.embed.params());
        p.extend(self.pos.params());
        for b in &self.blocks { p.extend(b.params()); }
        p.extend(self.final_norm.params());
        p.extend(self.head.params());
        p
    }

    /// Parameters with stable names, for checkpoints and optimizers.
    pub fn named_params(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        out.push(("embed.weight".into(), self.embed.weight.clone()));
        out.push(("pos.weight".into(), self.pos.weight.clone()));
        for (i, b) in self.blocks.iter().enumerate() {
            let p = |n: &str| format!("blocks.{i}.{n}");
            out.push((p("attn_norm"), b.attn_norm.weight.clone()));
            out.push((p("wq.weight"), b.attn.wq.weight.clone()));
            out.push((p("wq.bias"), b.attn.wq.bias.clone().unwrap()));
            out.push((p("wk.weight"), b.attn.wk.weight.clone()));
            out.push((p("wk.bias"), b.attn.wk.bias.clone().unwrap()));
            out.push((p("wv.weight"), b.attn.wv.weight.clone()));
            out.push((p("wv.bias"), b.attn.wv.bias.clone().unwrap()));
            out.push((p("wo.weight"), b.attn.wo.weight.clone()));
            out.push((p("wo.bias"), b.attn.wo.bias.clone().unwrap()));
            out.push((p("ffn_norm"), b.ffn_norm.weight.clone()));
            out.push((p("gate.weight"), b.gate.weight.clone()));
            out.push((p("gate.bias"), b.gate.bias.clone().unwrap()));
            out.push((p("up.weight"), b.up.weight.clone()));
            out.push((p("up.bias"), b.up.bias.clone().unwrap()));
            out.push((p("down.weight"), b.down.weight.clone()));
            out.push((p("down.bias"), b.down.bias.clone().unwrap()));
        }
        out.push(("final_norm".into(), self.final_norm.weight.clone()));
        out.push(("head.weight".into(), self.head.weight.clone()));
        out.push(("head.bias".into(), self.head.bias.clone().unwrap()));
        out
    }

    /// Temperature sampling with a deterministic xorshift RNG.
    pub fn generate_sample(&self, prompt: &[usize], steps: usize, temperature: f32, seed: u64) -> Vec<usize> {
        let mut ids = prompt.to_vec();
        let mut rng = seed | 1;
        for _ in 0..steps {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            let logits = self.forward(&ids).values();
            let c = logits.len() / ids.len();
            let last = &logits[(ids.len() - 1) * c..ids.len() * c];
            let mx = last.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut probs: Vec<f32> = last.iter().map(|x| ((x - mx) / temperature).exp()).collect();
            let sum: f32 = probs.iter().sum();
            for p in &mut probs { *p /= sum; }
            let r = ((rng >> 11) as f32 / ((1u64 << 53) as f32)) * sum;
            let mut acc = 0.0;
            let mut chosen = c - 1;
            for (i, p) in probs.iter().enumerate() {
                acc += *p;
                if acc >= r { chosen = i; break; }
            }
            ids.push(chosen);
        }
        ids
    }

    /// Greedy sampling from a prompt; returns token ids.
    pub fn generate(&self, prompt: &[usize], steps: usize) -> Vec<usize> {
        let mut ids = prompt.to_vec();
        for _ in 0..steps {
            let logits = self.forward(&ids);
            let v = logits.values();
            let c = v.len() / ids.len();
            let last = &v[(ids.len() - 1) * c..ids.len() * c];
            let mut best = 0;
            for i in 1..c { if last[i] > last[best] { best = i; } }
            ids.push(best);
        }
        ids
    }
}
