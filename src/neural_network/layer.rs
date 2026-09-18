pub struct Layer {
    in_features: usize,
    out_features: usize,
    weights: Vec<Vec<f32>>,
    bias: Vec<f32>
}