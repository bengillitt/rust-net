
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
    result: FlatMatrix,
}

impl Layer {
    pub fn new(in_features: usize, out_features: usize, activation: ActivationFunction) -> Layer {
        return Layer{
            in_features: in_features,
            out_features: out_features,
            weights: FlatMatrix{mat: vec![0.0; in_features * out_features], rows: out_features},
            bias: FlatMatrix{mat: vec![0.0; out_features], rows: 1},
            activation: activation,
            result: FlatMatrix{mat: vec![], rows: 0},
        };
    }

    pub fn pass(&mut self, inputs: &FlatMatrix) -> FlatMatrix {
        self.result = matrix_math::matrix_activation(&matrix_math::matrix_add_bias_flat(&matrix_math::matrix_multiply_flat(inputs, &self.weights, true).unwrap(), &self.bias).unwrap(), &self.activation);

        return self.result.clone();
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

    fn derivative(&self, a: f32) -> f32 {
        match self.activation() {
            ActivationFunction::Sigmoid => a * (1.0 - a),
            ActivationFunction::Linear => 1.0,
            ActivationFunction::ReLU => if a > 0.0 { 1.0 } else { 0.0 },
            ActivationFunction::Tanh => 1.0 - a.powi(2),
            _ => panic!("Unsupported activation function"),
        }
    }

    pub fn activation_derivative(&self) -> FlatMatrix {
        let out = self.result.mat.iter().map(|a| self.derivative(*a)).collect();

        return FlatMatrix{mat: out, rows: self.result.rows};
    }

    pub fn set_weights(&mut self, new_weights: &FlatMatrix) {
        self.weights = new_weights.clone();
    }

    pub fn set_bias(&mut self, new_bias: &FlatMatrix) {
        self.bias = new_bias.clone();
    }
}