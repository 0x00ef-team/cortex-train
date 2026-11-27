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
- Optimizer: AdamW with decoupled weight decay.
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

Gradient checks against finite differences cover `matmul`, `cross_entropy`
and `rms_norm`.

## Roadmap

- [ ] CUDA autograd backend (reuse `cortex` CUDA kernels for forward/backward)
- [ ] GGUF export so trained weights can be loaded directly by `cortex`
- [ ] More losses (MSE, BCE), optimizers (SGD, RMSProp), data loaders
- [ ] Checkpoint save/load

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
- Оптимизатор AdamW с decoupled weight decay.
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

Градиенты `matmul`, `cross_entropy` и `rms_norm` сверены с конечными разностями.

### Планы

- CUDA-автодиф (переиспользовать CUDA-ядра `cortex`)
- экспорт в GGUF, чтобы обученную модель грузил `cortex`
- MSE/BCE, SGD/RMSProp, загрузчики данных, чекпоинты

Лицензия: MIT OR Apache-2.0.
