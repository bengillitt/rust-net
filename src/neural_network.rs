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
    pub fn forward_pass(&self, inputs: FlatMatrix) -> FlatMatrix {
        let mut input = inputs;

        for i in 0..self.layers.len() {
            input = self.layers[i].pass(&input);
        }

        if self.softmax_enabled {
            input = matrix_math::matrix_softmax(&input);
        }

        return input;
    }

    pub fn back_prop(&self, inputs: FlatMatrix, expected_outputs: FlatMatrix) {
        let outputs = self.forward_pass(inputs);

        let loss = self.calculate_loss(outputs, expected_outputs);

        for i in self.layers.iter().rev() {
            
        }
    }

    fn calculate_deltas(&self, layer: &mut Layer, ) {}

    fn calculate_loss(&self, outputs: FlatMatrix, expected_outputs: FlatMatrix) -> Result<FlatMatrix, String> {
        if outputs.rows != expected_outputs.rows || outputs.cols() != expected_outputs.cols() {
            return Err("Matrix orders differ".to_string());
        }

        let out: Vec<f32> = outputs.mat.iter().zip(expected_outputs.mat).map(|(a, b)| f32::powi(a - b, 2)).collect();

        return Ok(FlatMatrix { mat: out, rows: outputs.rows });
    }
}