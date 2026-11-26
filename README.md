# rust-net

A feedforward neural network built completely from scratch in Rust, implementing Stochastic Gradient Descent (SGD) and backpropagation without external machine learning libraries.

## Overview

This project was built to deeply understand the underlying mathematics and memory mechanics of neural networks. Rather than relying on high-level frameworks like PyTorch or TensorFlow, all matrix operations, activation functions, loss calculations, and gradient updates are implemented natively in Rust.

## Features

* **Zero ML Frameworks:** Built using pure Rust (and standard math/linear algebra primitives) with no autodiff engines.
* **Custom Backpropagation:** Manual derivation and implementation of the chain rule across layers to compute weight and bias gradients.
* **Stochastic Gradient Descent (SGD):** Configurable learning rates, batching, and training epochs.
* **Modular Architecture:** Easily define custom layer dimensions and network depths.

## How It Works

1. **Forward Pass:** Inputs are propagated through linear layers ($Z = WX + b$) followed by non-linear activation functions.
2. **Loss Calculation:** Computes prediction error against ground-truth labels.
3. **Backward Pass:** Calculates partial derivatives of the loss with respect to weights and biases layer-by-layer from output to input.
4. **Parameter Update:** Adjusts weights and biases in the opposite direction of the gradient ($\theta = \theta - \eta \nabla_\theta J(\theta)$).

## Getting Started

### Prerequisites

* [Rust & Cargo](https://www.rust-lang.org/tools/install) (latest stable version)

### Installation & Usage

Clone the repository and run the project in release mode for optimal matrix computation speed:

```bash
git clone [https://github.com/bengillitt/rust-net.git](https://github.com/bengillitt/rust-net.git)
cd rust-net
cargo run --release
