//! Cortex Train — general-purpose neural networks from scratch in Rust.
//!
//! No dependencies: dense tensors, reverse-mode autograd, composable modules
//! (Linear, Embedding, RMSNorm, causal attention, transformer blocks) and
//! AdamW. Create any architecture with `Transformer::new` or by composing
//! modules yourself, then train with `loss.backward()` + `optimizer.step()`.

pub mod arch;
pub mod checkpoint;
pub mod cuda;
pub mod cuda_model;
pub mod corpus;
pub mod data;
pub mod distributed;
pub mod f16;
pub mod gguf_export;
pub mod loss;
pub mod nn;
pub mod optim;
pub mod tensor;
pub mod thermal;

pub use nn::{CausalAttention, Embedding, Linear, RMSNorm, Transformer, TransformerBlock};
pub use optim::{AdamW, RMSProp, SGD};
pub use tensor::Tensor;
pub use arch::{preset, CortexAiPreset, CORTEX_AI_120M, CORTEX_AI_1B, CORTEX_AI_300M, CORTEX_AI_8M};
pub use corpus::Corpus;
pub use cuda_model::{GpuConfig, GpuTransformer};
pub use data::{Dataset, DataLoader};
pub use f16::{f32_to_half, half_to_f32};
pub use gguf_export::{export as export_gguf, ExportOptions};
pub use loss::{binary_cross_entropy, mse};
