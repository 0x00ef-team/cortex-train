# Cortex Train

[![Rust](https://img.shields.io/badge/Rust-1.80%2B-orange.svg)](https://rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)

**General-purpose neural networks from scratch in Rust.** No dependencies:
dense tensors, reverse-mode autograd, composable modules and AdamW. You decide
the architecture — there is no hard-coded “tiny LLaMA” path.

This repository is attached to `cortex-llm` as a git submodule at
`crates/cortex-train`; it can also be used standalone.

## What is included

- `Tensor`: shape, f32 storage, parameter flag, deterministic xorshift init.
- Autograd tape: `add`, `add_scalar`, `scale`, `mul`, `matmul` (2D and batched
  3D), `transpose`, `transpose01`, `transpose12`, `reshape`, `softmax`, `silu`,
  `tanh`, `sum`, `embedding`, `rms_norm`, `cross_entropy`, `backward()`.
- Modules: `Linear`, `Embedding`, `RMSNorm`, `CausalAttention`,
  `TransformerBlock`, `Transformer` (decoder-only, any vocab / dim / layers /
  heads / hidden size).
- Losses: cross-entropy, MSE, binary cross-entropy.
- Optimizers: AdamW (decoupled weight decay), SGD+momentum, RMSProp.
- Data: `Dataset` (next-token windows), `DataLoader` (deterministic shuffling).
- Checkpoints: named tensor save/load (`CTRK1` binary format).
- Greedy and temperature sampling.

## Create and train any model

```rust
use cortex_train::{AdamW, Transformer};

let vocab = 64;
let model = Transformer::new(vocab, 48, 3, 3, 128, 42);
let params = model.params();
let mut opt = AdamW::new(params, 0.005).with_weight_decay(0.001);

let ids = vec![1, 2, 3, 4, 5, 6];
let targets = vec![2, 3, 4, 5, 6, 7];

for _ in 0..100 {
    opt.zero_grad();
    let loss = model.loss(&ids, &targets);
    loss.backward();
    opt.step();
}

let out = model.generate_sample(&ids[..3], 32, 0.8, 777);
```

Or compose your own architecture from `Linear`, `Embedding`, `RMSNorm` and
`CausalAttention` — the autograd tape handles any DAG of these ops.

## Run the example

```bash
cargo run --release --example train
```

It builds a character-level transformer from scratch on a tiny corpus,
trains it with AdamW (loss goes from ~3.5 to ~0.5–1.0), then samples text.
The training loop is generic — change `Transformer::new(...)` to create a
different model.

## Tests

```bash
cargo test
```

7 tests: finite-difference gradient checks for `matmul`, `cross_entropy`,
`rms_norm`, MSE/BCE, dataset/loader behavior, all three optimizers and a
checkpoint round-trip.

## CUDA backward

`src/cuda.rs` is a dependency-free CUDA driver/NVRTC layer. Run:

```bash
cargo run --release --example cuda_train
# step  40 loss 0.306073
# step 200 loss 0.000000

cargo run --release --example cuda_transformer_multi
# trains on datasets/code.txt (The Stack smol-xs subset)
# 6-layer / dim 128 model on real code, lr 5e-5:
# step 250 loss 4.20 -> step 2000 loss ~3.4-4.4 (batch loss varies)
```

It trains `y = W·x` entirely on device with compiled backward kernels.

`examples/cuda_transformer_train` goes further: one full decoder block
(embedding, RMSNorm, causal attention, SwiGLU FFN, residuals, output head)
trains end-to-end on GPU:

```bash
cargo run --release --example cuda_transformer_train
# step  10 loss 3.0442  ->  step 100 loss 0.8151
```

All backward passes (attention, softmax, RMSNorm, SiLU, embedding) are
compiled CUDA kernels. `save_checkpoint` exports host-readable `CTRK1` weights.

Honest note: a 1.5M-parameter model trained for ~2000 short windows on 10 MB of
code learns token statistics but does not yet generate useful code — greedy
sampling degenerates to whitespace. Scaling needs more layers, longer context,
more data and more steps.

## Roadmap

- [x] CUDA device layer: NVRTC kernels for matmul forward/backward, MSE
      forward/backward and AdamW; `examples/cuda_train` trains a layer on GPU
- [x] GGUF export: `export_gguf(path, model, ExportOptions)` writes an F32
      LLaMA-style GGUF that `cortex` loads and generates from
- [x] MSE/BCE, SGD/RMSProp, data loaders, checkpoint save/load
- [x] CUDA training backend: explicit forward/backward kernels for MLP and
      multi-layer transformer (`GpuTransformer`), AdamW, gradient clipping,
      thermal guard. The CPU tape and GPU device layer remain separate APIs.

## License

MIT OR Apache-2.0.

---

# Cortex Train (RU)

**Обучение нейросетей с нуля на Rust.** Без зависимостей: плотные тензоры,
автодифференцирование, компонуемые модули и AdamW. Архитектуру задаёшь ты —
жёстко заданной «tiny LLaMA» здесь нет.

Репозиторий подключён к `cortex-llm` как git submodule по пути
`crates/cortex-train`, но работает и отдельно.

### Что есть

- `Tensor`: форма, f32-данные, флаг параметра, детерминированная инициализация.
- Autograd-lenta: `add`, `scale`, `mul`, `matmul` (2D и батчевые 3D),
  `transpose`, `reshape`, `softmax`, `silu`, `tanh`, `sum`, `embedding`,
  `rms_norm`, `cross_entropy`, `backward()`.
- Модули: `Linear`, `Embedding`, `RMSNorm`, `CausalAttention`,
  `TransformerBlock`, `Transformer` — любой словарь/размерность/слои/головы.
- Функции потерь: cross-entropy, MSE, BCE.
- Оптимизаторы: AdamW, SGD+momentum, RMSProp.
- Данные: `Dataset` (окна next-token), `DataLoader` (детерминированный shuffle).
- Чекпоинты: именованные тензоры в бинарном формате `CTRK1`.
- Greedy и temperature-сэмплинг.

### Пример

```bash
cargo run --release --example train
```

Создаётся.char-level трансформер, обучается на небольшом тексте (loss падает
с ~3.5 до ~0.5–1.0), затем генерируется текст. Цикл обучения общий — меняй
`Transformer::new(...)`, чтобы получить другую модель.

### Тесты

```bash
cargo test
```

7 тестов: градиенты `matmul`, `cross_entropy`, `rms_norm`, MSE/BCE, работа
`Dataset`/`DataLoader`, все три оптимизатора и round-trip чекпоинта.

### Планы

- [x] CUDA-обучение: явные ядра forward/backward для MLP и многослойного
      трансформера, AdamW, клиппинг градиентов, thermal guard
- [x] экспорт в GGUF: `export_gguf(...)` пишет F32-файл, который грузит `cortex`
- MSE/BCE, SGD/RMSProp, загрузчики данных, чекпоинты

Лицензия: MIT OR Apache-2.0.

## GGUF export

```rust
use cortex_train::{export_gguf, ExportOptions, Transformer};

let model = Transformer::new(32, 16, 2, 4, 32, 7);
let tokens: Vec<Vec<u8>> = /* one byte-string per vocab id */;
export_gguf("/tmp/model.gguf", &model, ExportOptions {
    tokens: &tokens, context: 128, name: "my-model",
})?;
```

Verified: `cortex/examples/info` reads the file and `cortex/examples/generate`
produces tokens from it. Learned positional embeddings are preserved by using
a very large RoPE base (rotation becomes identity).
