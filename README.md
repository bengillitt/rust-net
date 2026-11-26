# rust-net

A feedforward neural network built completely from scratch in Rust — forward pass, backpropagation, SGD, softmax/cross-entropy and weight decay — with no ML frameworks or autodiff. The only dependency is `rand`.

The default experiment trains the network on **modular addition** — predicting `(x + y) mod 97` — the same toy task used in the *grokking* literature (Power et al., 2022) and in mechanistic interpretability work on how networks learn algorithms (Nanda et al., 2023).

## Why I built it

I wanted to understand what frameworks like PyTorch actually do under the hood: every matrix operation, activation, loss and gradient here is written by hand, including deriving the chain rule for each layer. Modular addition is a small, fully-known task, which makes it a good testbed for studying *how* a network learns — memorisation vs. generalisation — rather than just whether it does.

## Features

- **Zero ML frameworks** — pure Rust, no autodiff; gradients derived by hand.
- **Configurable architecture** — choose depth, layer widths and activation per layer at runtime.
- **Activations:** ReLU, Sigmoid, Tanh, Linear, plus an optional softmax output.
- **Losses:** cross-entropy (with softmax) or mean squared error.
- **Mini-batch SGD** (batch size 32) with optional **weight decay** (decoupled, applied after each step).
- **Sensible initialisation** — He initialisation for ReLU layers, Xavier/Glorot for everything else.
- **Numerically stable softmax** (max-subtraction before exponentiating).

## Performance

All matrices are stored as flat, contiguous `Vec<f32>` buffers (row-major) rather than nested vectors, so data is cache-friendly and the hot loops are easy for the compiler to vectorise:

- **Matrix multiply** uses `i-k-j` loop ordering, so the innermost loop streams contiguously through memory — the pattern LLVM auto-vectorises into SIMD instructions in `--release` builds.
- **Transpose-aware multiply** — a dedicated path computes `A · Bᵀ` directly by dotting rows, avoiding materialising the transpose in the forward pass.
- **Cache-blocked transpose** (32×32 tiles) for the backward pass.

<!-- TODO (Ben): add a before/after number here, e.g. naive i-j-k loop vs current version,
     or confirm the vector instructions with `cargo asm` / godbolt.org -->

## Results

Modular addition, p = 97 → 9,409 possible `(x, y)` pairs, 50/50 train/test split.

| Setting | Value |
|---|---|
| Architecture | 194 (one-hot x ‖ one-hot y) → 128 ReLU → 97 softmax |
| Optimiser | Mini-batch SGD, lr = 0.1, batch size 32 |
| Weight decay | 0.001 |
| Training | 1,000 epochs |

| Epochs | Train accuracy | Test accuracy |
|---|---|---|
| 100 | 6.1% | 0.0% |
| 300 | 63.5% | 0.1% |
| 500 | 99.3% | 7.7% |
| 1,000 | 99.98% | 16.4% |

The network **memorises the training set almost perfectly, while test accuracy lags far behind and only creeps upward slowly** — the classic memorisation-before-generalisation pattern seen in grokking. A stronger weight decay (0.01) prevented learning altogether with plain SGD, so the regularisation strength matters a lot.

### Next steps

- Train much longer (and try AdamW) to see whether test accuracy eventually jumps — i.e. whether the network fully *groks*.
- Log the weight norm over training alongside accuracy.
- Look inside the trained network for the Fourier/trigonometric structure that Nanda et al. found networks use to implement modular addition.

## Getting started

Requires [Rust & Cargo](https://www.rust-lang.org/tools/install) (latest stable).

```bash
git clone https://github.com/bengillitt/rust-net.git
cd rust-net
cargo run --release
```

The program asks for the architecture interactively. To reproduce the results above, answer:

```
How many layers?            2
How many inputs?            194
Softmax?                    true
Neurons for layer 1:        128   → activation 2 (ReLU)
Neurons for output layer:   97    → activation 4 (Linear; softmax is applied on top)
Weight decay?               true  → 0.001
```

Or pipe the answers in:

```bash
printf '2\n194\ntrue\n128\n2\n97\n4\ntrue\n0.001\n' | cargo run --release
```

## How it works

1. **Forward pass:** each layer computes `Z = XWᵀ + b`, then applies its activation; softmax is applied to the final layer if enabled.
2. **Loss:** cross-entropy (softmax) or mean squared error against one-hot targets.
3. **Backward pass:** the output delta is `ŷ − y` for softmax + cross-entropy; earlier deltas are propagated as `δₗ = (δₗ₊₁ · Wₗ₊₁) ⊙ f′(aₗ)`.
4. **Update:** `W ← W − (η / batch) · δᵀX`, biases likewise, then weight decay scales `W ← W · (1 − λη)`.

## Project structure

```
src/
├── main.rs                  # dataset generation, training loop, evaluation
├── matrix_math.rs           # FlatMatrix + all matrix ops (matmul, transpose, softmax, activations)
├── neural_network.rs        # Network: forward pass, backprop, loss, weight decay
└── neural_network/layer.rs  # Layer: weights, biases, initialisation, gradient updates
```

## References

- Power et al. (2022), *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets* — https://arxiv.org/abs/2201.02177
- Nanda et al. (2023), *Progress Measures for Grokking via Mechanistic Interpretability* — https://arxiv.org/abs/2301.05217
