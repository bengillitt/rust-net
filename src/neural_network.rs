pub mod layer;

pub use super::matrix_math;

use matrix_math::FlatMatrix;

use layer::Layer;

pub struct Network {
    layers: Vec<Layer>,
    softmax_enabled: bool,
    weight_decay: Option<f64>,
    matrix: matrix_math::FlatMatrix,
}

impl Network {
    pub fn forward_pass(&mut self, inputs: FlatMatrix) -> FlatMatrix {
        let mut input = inputs;

        for i in 0..self.layers.len() {
            input = self.layers[i].pass(&input);
        }

        if self.softmax_enabled {
            input = matrix_math::matrix_softmax(&input);
        }

        return input;
    }

    pub fn back_prop(&mut self, inputs: FlatMatrix, expected_outputs: FlatMatrix) {
        let outputs = self.forward_pass(inputs);

        let loss = self.calculate_loss(outputs, expected_outputs);

        let next_delta = loss.unwrap();

        for i in self.layers.iter().rev() {
            
        }
    }

    fn calculate_deltas(&self, layer: &Layer, next_delta: &FlatMatrix) -> FlatMatrix {
        // delta_prev = delta*W hadamard product with the derivative of the activation function
        // Hadamard Product - multiply every element in the matrix by the operand

        let multiply = matrix_math::matrix_multiply_flat(next_delta, layer.weights(), false).unwrap();
        let derivatives = layer.activation_derivative();

        let out = matrix_math::matrix_hadamard_product(&multiply, &derivatives).unwrap();

        return out;
    }

    fn calculate_loss(&self, outputs: FlatMatrix, expected_outputs: FlatMatrix) -> Result<FlatMatrix, String> {
        if outputs.rows != expected_outputs.rows || outputs.cols() != expected_outputs.cols() {
            return Err("Matrix orders differ".to_string());
        }

        let out: Vec<f32> = outputs.mat.iter().zip(expected_outputs.mat).map(|(a, b)| f32::powi(a - b, 2)).collect();

        return Ok(FlatMatrix { mat: out, rows: outputs.rows });
    }
}