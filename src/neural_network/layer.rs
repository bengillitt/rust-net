
use super::super::matrix_math::{FlatMatrix};
use super::super::matrix_math;

#[derive(PartialEq)]
pub enum ActivationFunction {
    Sigmoid,
    ReLU,
    Tanh,
    Linear,
}

pub struct Layer {
    in_features: usize,
    out_features: usize,
    weights: FlatMatrix,
    bias: FlatMatrix,
    activation: ActivationFunction,
}

impl Layer {
    pub fn new(in_features: usize, out_features: usize, activation: ActivationFunction) -> Layer {
        return Layer{
            in_features: in_features,
            out_features: out_features,
            weights: FlatMatrix{mat: vec![0.0; in_features * out_features], rows: in_features},
            bias: FlatMatrix{mat: vec![0.0; out_features], rows: 1},
            activation: activation
        };
    }

    pub fn pass(&self, inputs: &FlatMatrix) -> FlatMatrix {
        matrix_math::matrix_activation(&matrix_math::matrix_add_bias_flat(&matrix_math::matrix_multiply_flat(inputs, &self.weights).unwrap(), &self.bias).unwrap(), &self.activation)
    }

    pub fn in_features(&self) -> usize {
        return self.in_features;
    }

    pub fn out_features(&self) -> usize {
        return self.out_features;
    }

    pub fn weights(&self) -> &FlatMatrix {
        return &self.weights;
    }

    pub fn bias(&self) -> &FlatMatrix {
        return &self.bias;
    }

    pub fn activation(&self) -> &ActivationFunction {
        return &self.activation;
    }

    pub fn set_weights(&mut self, new_weights: &FlatMatrix) {
        self.weights = new_weights.clone();
    }

    pub fn set_bias(&mut self, new_bias: &FlatMatrix) {
        self.bias = new_bias.clone();
    }
}