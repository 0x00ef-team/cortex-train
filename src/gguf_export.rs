//! Export a trained CPU `Transformer` to a GGUF v3 file loadable by `cortex`.
//!
//! The native model uses learned positional embeddings; setting a very large
//! RoPE base makes Cortex's rotary transform effectively identity, so the
//! exported weights behave like the trained model.

use crate::Transformer;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

pub struct ExportOptions<'a> {
    pub tokens: &'a [Vec<u8>],
    pub context: usize,
    pub name: &'a str,
}

struct W { buf: Vec<u8> }
impl W {
    fn new() -> Self { Self { buf: Vec::new() } }
    fn u32(&mut self, v: u32) { self.buf.extend_from_slice(&v.to_le_bytes()); }
    fn u64(&mut self, v: u64) { self.buf.extend_from_slice(&v.to_le_bytes()); }
    fn f32(&mut self, v: f32) { self.buf.extend_from_slice(&v.to_le_bytes()); }
    fn bytes(&mut self, b: &[u8]) { self.buf.extend_from_slice(b); }
    fn str(&mut self, s: &[u8]) { self.u64(s.len() as u64); self.bytes(s); }
    fn key(&mut self, s: &str) { self.str(s.as_bytes()); }
    fn str_key(&mut self, k: &str, v: &[u8]) { self.key(k); self.u32(8); self.str(v); }
    fn u32_key(&mut self, k: &str, v: u32) { self.key(k); self.u32(4); self.u32(v); }
    fn f32_key(&mut self, k: &str, v: f32) { self.key(k); self.u32(6); self.f32(v); }
    fn bool_key(&mut self, k: &str, v: bool) { self.key(k); self.u32(7); self.buf.push(v as u8); }
}

/// Write a minimal F32 GGUF with the standard LLaMA tensor layout.
pub fn export(path: impl AsRef<Path>, model: &Transformer, opt: ExportOptions<'_>) -> std::io::Result<()> {
    let vocab = model.vocab();
    let dim = model.dim();
    let layers = model.layers();
    let heads = model.heads();
    let hidden = model.hidden();
    assert_eq!(vocab, opt.tokens.len(), "token list must match vocab size");
    assert_eq!(dim % heads, 0);

    let mut specs: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
    let push = |name: String, dims: Vec<usize>, t: &crate::Tensor, specs: &mut Vec<_>| {
        specs.push((name, dims, t.values()));
    };
    push("token_embd.weight".into(), vec![dim, vocab], &model.embed.weight, &mut specs);
    for (i, b) in model.blocks.iter().enumerate() {
        let p = |s: &str| format!("blk.{i}.{s}");
        push(p("attn_norm.weight"), vec![dim], &b.attn_norm.weight, &mut specs);
        push(p("attn_q.weight"), vec![dim, dim], &b.attn.wq.weight.transpose(), &mut specs);
        push(p("attn_k.weight"), vec![dim, dim], &b.attn.wk.weight.transpose(), &mut specs);
        push(p("attn_v.weight"), vec![dim, dim], &b.attn.wv.weight.transpose(), &mut specs);
        push(p("attn_output.weight"), vec![dim, dim], &b.attn.wo.weight.transpose(), &mut specs);
        push(p("ffn_norm.weight"), vec![dim], &b.ffn_norm.weight, &mut specs);
        push(p("ffn_gate.weight"), vec![dim, hidden], &b.gate.weight.transpose(), &mut specs);
        push(p("ffn_up.weight"), vec![dim, hidden], &b.up.weight.transpose(), &mut specs);
        push(p("ffn_down.weight"), vec![hidden, dim], &b.down.weight.transpose(), &mut specs);
    }
    push("output_norm.weight".into(), vec![dim], &model.final_norm.weight, &mut specs);
    push("output.weight".into(), vec![dim, vocab], &model.head.weight.transpose(), &mut specs);

    let mut w = W::new();
    w.bytes(b"GGUF");
    w.u32(3);
    w.u64(specs.len() as u64);
    w.u64(20); // metadata count

    w.str_key("general.architecture", b"llama");
    w.str_key("general.name", opt.name.as_bytes());
    w.u32_key("llama.block_count", layers as u32);
    w.u32_key("llama.embedding_length", dim as u32);
    w.u32_key("llama.feed_forward_length", hidden as u32);
    w.u32_key("llama.attention.head_count", heads as u32);
    w.u32_key("llama.attention.head_count_kv", heads as u32);
    w.f32_key("llama.attention.layer_norm_rms_epsilon", 1e-5);
    w.u32_key("llama.context_length", opt.context as u32);
    w.f32_key("llama.rope.freq_base", 1e30); // effectively disables rotation
    w.u32_key("llama.vocab_size", vocab as u32);
    w.str_key("tokenizer.ggml.model", b"llama");
    w.key("tokenizer.ggml.tokens"); w.u32(9); w.u32(8); w.u64(vocab as u64);
    for t in opt.tokens { w.str(t); }
    w.key("tokenizer.ggml.scores"); w.u32(9); w.u32(6); w.u64(vocab as u64);
    for i in 0..vocab { w.f32(i as f32 * 0.01); }
    w.key("tokenizer.ggml.token_type"); w.u32(9); w.u32(4); w.u64(vocab as u64);
    for _ in 0..vocab { w.u32(1); }
    w.u32_key("tokenizer.ggml.bos_token_id", 1);
    w.u32_key("tokenizer.ggml.eos_token_id", 2);
    w.u32_key("tokenizer.ggml.unknown_token_id", 0);
    w.bool_key("tokenizer.ggml.add_bos_token", false);
    w.bool_key("tokenizer.ggml.add_eos_token", false);

    let mut running = 0usize;
    for (name, dims, _) in &specs {
        w.str(name.as_bytes());
        w.u32(dims.len() as u32);
        for d in dims { w.u64(*d as u64); }
        w.u32(0); // F32
        w.u64(running as u64);
        running += dims.iter().product::<usize>() * 4;
    }
    let pad = (32 - w.buf.len() % 32) % 32;
    w.buf.resize(w.buf.len() + pad, 0);
    for (_, _, values) in &specs {
        for &v in values { w.f32(v); }
    }

    let mut out = BufWriter::new(File::create(path)?);
    out.write_all(&w.buf)?;
    out.flush()
}
