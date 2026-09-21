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
            
        }

        return input;
    }
}