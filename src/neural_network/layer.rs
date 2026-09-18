pub struct Layer {
    in_features: usize,
    out_features: usize,
    weights: Vec<f32>,
    bias: Vec<f32>
}

impl Layer {
    pub fn new(in_features: usize, out_features: usize) -> Layer {
        return Layer{
            in_features: in_features,
            out_features: out_features,
            weights: vec![0.0; in_features * out_features],
            bias: vec![0.0; out_features],
        };
    }

    pub fn in_features(&self) -> usize {
        return self.in_features;
    }

    pub fn out_features(&self) -> usize {
        return self.out_features;
    }

    pub fn weights(&self) -> &Vec<f32> {
        return &self.weights;
    }

    pub fn bias(&self) -> &Vec<f32> {
        return &self.bias;
    }

    pub fn set_weights(&mut self, new_weights: Vec<f32>) {
        self.weights = new_weights;
    }

    pub fn set_bias(&mut self, new_bias: Vec<f32>) {
        self.bias = new_bias;
    }
}