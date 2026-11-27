//! Cortex Train — general-purpose neural networks from scratch in Rust.
//!
//! No dependencies: dense tensors, reverse-mode autograd, composable modules
//! (Linear, Embedding, RMSNorm, causal attention, transformer blocks) and
//! AdamW. Create any architecture with `Transformer::new` or by composing
//! modules yourself, then train with `loss.backward()` + `optimizer.step()`.

pub mod nn;
pub mod optim;
pub mod tensor;

pub use nn::{CausalAttention, Embedding, Linear, RMSNorm, Transformer, TransformerBlock};
pub use optim::AdamW;
pub use tensor::{cross_entropy, Tensor};
