mod layer;

use super::matrix_math;

use layer::Layer;

pub struct Network {
    layers: Vec<Layer>,
    softmax_enabled: bool,
    weight_decay: Option<f64>
}