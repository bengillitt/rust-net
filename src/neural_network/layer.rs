
use super::super::matrix_math::{FlatMatrix};
use super::super::matrix_math;
use rand;

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
    inputs: FlatMatrix,
    result: FlatMatrix,
    deltas: FlatMatrix,
}

impl Layer {
    pub fn new(in_features: usize, out_features: usize, activation: ActivationFunction) -> Layer {
        let mut weights = vec![];

        let xavier_limit = f32::sqrt(6.0 / ((in_features + out_features) as f32));
        let he_limit = f32::sqrt(6.0 / (in_features as f32));

        for _ in 0..in_features*out_features {
            if activation == ActivationFunction::ReLU {
                weights.push((2.0 * rand::random::<f32>() - 1.0) * he_limit);
            } else {
                weights.push((2.0 * rand::random::<f32>() - 1.0) * xavier_limit);
            }
        }

        return Layer{
            in_features: in_features,
            out_features: out_features,
            weights: FlatMatrix{mat: weights, rows: out_features},
            bias: FlatMatrix{mat: vec![0.0; out_features], rows: 1},
            activation: activation,
            inputs: FlatMatrix{mat: vec![], rows: 0},
            result: FlatMatrix{mat: vec![], rows: 0},
            deltas: FlatMatrix{mat: vec![0.0; in_features * out_features], rows: out_features},
        };
    }

    pub fn pass(&mut self, inputs: &FlatMatrix) -> FlatMatrix {
        self.inputs = inputs.clone();
        self.result = matrix_math::matrix_activation(&matrix_math::matrix_add_bias_flat(&matrix_math::matrix_multiply_flat(inputs, &self.weights, true).unwrap(), &self.bias).unwrap(), &self.activation);

        return self.result.clone();
    }

    pub fn set_deltas(&mut self, new_deltas: FlatMatrix) {
        self.deltas = new_deltas;
    }

    pub fn delta_weights(&self) -> FlatMatrix {
        let delta_t = matrix_math::transpose_mat(&self.deltas).unwrap();

        return matrix_math::matrix_multiply_flat(&delta_t, &self.inputs, false).unwrap();
    }

    pub fn update_weights(&mut self, learning_rate: f32) {
        self.weights = matrix_math::matrix_subtract_flat(&self.weights, &matrix_math::matrix_scalar_multiply(&self.delta_weights(), learning_rate / self.inputs.rows as f32)).unwrap();

        let batch_size = self.deltas.rows;
        let scale = learning_rate / batch_size as f32;

        for j in 0..self.out_features {
            let mut gradient = 0.0;

            for b in 0..batch_size {
                gradient += self.deltas.mat[b * self.out_features + j];
            }

            self.bias.mat[j] -= scale * gradient;
        }
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