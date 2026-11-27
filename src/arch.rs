//! Ready-made architecture presets for Cortex-AI models.
//!
//! The same `Transformer` implementation powers every preset. A 1B config is
//! provided for scale; training it from scratch is a cluster-scale job, so
//! smaller presets exist for real experiments on a single machine.

use crate::Transformer;

#[derive(Clone, Copy, Debug)]
pub struct CortexAiPreset {
    pub name: &'static str,
    pub vocab: usize,
    pub dim: usize,
    pub layers: usize,
    pub heads: usize,
    pub hidden: usize,
    pub ctx: usize,
    pub seed: u64,
}

pub const CORTEX_AI_1B: CortexAiPreset = CortexAiPreset {
    name: "cortex-ai-1b",
    vocab: 32_000,
    dim: 2048,
    layers: 18,
    heads: 16,
    hidden: 8192,
    ctx: 4096,
    seed: 1,
};

pub const CORTEX_AI_300M: CortexAiPreset = CortexAiPreset {
    name: "cortex-ai-300m",
    vocab: 8192,
    dim: 1024,
    layers: 16,
    heads: 16,
    hidden: 4096,
    ctx: 1024,
    seed: 4,
};

pub const CORTEX_AI_120M: CortexAiPreset = CortexAiPreset {
    name: "cortex-ai-120m",
    vocab: 4096,
    dim: 512,
    layers: 8,
    heads: 8,
    hidden: 2048,
    ctx: 1024,
    seed: 2,
};

pub const CORTEX_AI_8M: CortexAiPreset = CortexAiPreset {
    name: "cortex-ai-8m",
    vocab: 1024,
    dim: 256,
    layers: 8,
    heads: 4,
    hidden: 1024,
    ctx: 512,
    seed: 3,
};

impl CortexAiPreset {
    /// Approximate trainable parameter count (with untied output head).
    pub fn params(&self) -> usize {
        let embed = self.vocab * self.dim;
        let pos = self.ctx * self.dim;
        let attn = 4 * self.dim * self.dim;
        let ffn = 3 * self.dim * self.hidden;
        let block = 2 * self.dim + attn + 4 * self.dim + ffn + 2 * self.dim; // norms + biases
        embed + pos + self.layers * block + self.dim + embed + self.vocab
    }

    pub fn build(&self) -> Transformer {
        Transformer::new(self.vocab, self.dim, self.layers, self.heads, self.hidden, self.seed)
    }
}

pub fn preset(name: &str) -> CortexAiPreset {
    match name {
        "1b" | "cortex-ai-1b" => CORTEX_AI_1B,
        "300m" | "cortex-ai-300m" => CORTEX_AI_300M,
        "120m" | "cortex-ai-120m" => CORTEX_AI_120M,
        "8m" | "cortex-ai-8m" => CORTEX_AI_8M,
        other => panic!("unknown preset: {other}"),
    }
}
