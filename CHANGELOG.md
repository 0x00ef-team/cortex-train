# Changelog

## 0.1.0

- Dense f32 tensors and reverse-mode autograd (no dependencies).
- Composable modules: Linear, Embedding, RMSNorm, CausalAttention,
  TransformerBlock, Transformer.
- Losses: cross-entropy, MSE, BCE.
- Optimizers: AdamW, SGD+momentum, RMSProp.
- Dataset/DataLoader and named checkpoint save/load (`CTRK1`).
- Greedy and temperature sampling.
- 7 tests including finite-difference gradient checks.
