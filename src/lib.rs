//! Cortex Train — general-purpose neural networks from scratch in Rust.
//!
//! No dependencies: dense tensors, reverse-mode autograd, composable modules
//! (Linear, Embedding, RMSNorm, causal attention, transformer blocks) and
//! AdamW. Create any architecture with `Transformer::new` or by composing
//! modules yourself, then train with `loss.backward()` + `optimizer.step()`.

pub mod checkpoint;
pub mod data;
pub mod loss;
pub mod nn;
pub mod optim;
pub mod tensor;

pub use nn::{CausalAttention, Embedding, Linear, RMSNorm, Transformer, TransformerBlock};
pub use optim::{AdamW, RMSProp, SGD};
pub use tensor::Tensor;
pub use data::{Dataset, DataLoader};
pub use loss::{binary_cross_entropy, mse};
