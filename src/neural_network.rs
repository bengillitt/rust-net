pub mod layer;

pub use super::matrix_math;

use matrix_math::FlatMatrix;

use layer::Layer;

pub struct Network {
    layers: Vec<Layer>,
    softmax_enabled: bool,
    weight_decay: Option<f64>,
    // matrix: matrix_math::FlatMatrix,
}

impl Network {
    pub fn new() -> Network {
        let mut layer_string = String::new();
        println!("How many layers?");
        std::io::stdin().read_line(&mut layer_string).expect("Failed to read line");

        let num_layers: u32 = layer_string.trim().parse().expect("Please type in a number");

        let mut inputs_string = String::new();
        println!("How many inputs?");
        std::io::stdin().read_line(&mut inputs_string).expect("Failed to read line");

        let num_inputs: usize = inputs_string.trim().parse().expect("Please type in a number");

        let mut softmax_string = String::new();
        println!("Softmax? (true/false)");
        std::io::stdin().read_line(&mut softmax_string).expect("Failed to read line");

        let softmax_enabled: bool = match &softmax_string.trim()[..] {
            "true" => true,
            "false" => false,
            _ => panic!("Not true or false for softmax"),
        };


        let mut layers: Vec<Layer> = Vec::new();

        let mut prev_outputs = num_inputs;

        for i in 0..num_layers {
            let mut output_string = String::new();
            if i == num_layers - 1 {
                println!("How many neurons for output layer");
            } else {
                println!("How many neurons for layer {}", i + 1);
            }

            std::io::stdin().read_line(&mut output_string).expect("Failed to read line");

            let num_outputs: usize = output_string.trim().parse().expect("Please type in a number");

            println!("What activation do you want?");
            println!("1. Sigmoid");
            println!("2. ReLU");
            println!("3. Tanh");
            println!("4. Linear");

            let mut activation_string = String::new();

            std::io::stdin().read_line(&mut activation_string).expect("Failed to read line");

            let activation_type: usize = activation_string.trim().parse().expect("Please type in a number");

            let mut activation = match activation_type {
                1 => layer::ActivationFunction::Sigmoid,
                2 => layer::ActivationFunction::ReLU,
                3 => layer::ActivationFunction::Tanh,
                4 => layer::ActivationFunction::Linear,
                _ => panic!("Not an activation function"),
            };

            if i == num_layers - 1 && softmax_enabled{
                activation = layer::ActivationFunction::Linear;
            }

            layers.push(Layer::new(prev_outputs, num_outputs, activation));

            prev_outputs = num_outputs;
        }

        let mut weight_decay = String::new();
        println!("Weight Decay? (true/false)");
        std::io::stdin().read_line(&mut weight_decay).expect("Failed to read line");

        let weight_decay_enabled: bool = match &weight_decay.trim()[..] {
            "true" => true,
            "false" => false,
            _ => panic!("Not true or false for weight decay"),
        };

        let weight_decay: Option<f64>;

        if weight_decay_enabled {
            let mut weight_decay_string = String::new();
            println!("Enter weight decay value");
            std::io::stdin().read_line(&mut weight_decay_string).expect("Failed to read line");

            weight_decay = Some(inputs_string.trim().parse().expect("Please type in a number"));
        } else {
            weight_decay = None;
        }

        return Network { layers, softmax_enabled: softmax_enabled, weight_decay: weight_decay }
    }

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

    pub fn back_prop(&mut self, inputs: FlatMatrix, expected_outputs: FlatMatrix, learning_rate: f32) {
        let outputs = self.forward_pass(inputs.clone());

        let loss_matrix = self.calculate_loss(&outputs, &expected_outputs).unwrap();

        let divisor: f32;

        if self.softmax_enabled {
            divisor = inputs.rows as f32;
        } else {
            divisor = inputs.rows as f32 * outputs.cols() as f32;
        }

        let loss = matrix_math::matrix_sum(&loss_matrix) / divisor;

        println!("Current Loss: {}", loss);

        let loss_derivative = self.calculate_loss_derivative(&outputs, &expected_outputs);

        let mut next_delta = matrix_math::matrix_hadamard_product(&loss_derivative.unwrap(), &self.layers[&self.layers.len() - 1].activation_derivative()).unwrap();

        for i in (0..self.layers.len()).rev() {
            if i != (self.layers.len() - 1) {
                next_delta = self.calculate_deltas(&self.layers[i], &self.layers[i+1], &next_delta);
            }

            self.layers[i].set_deltas(next_delta.clone());
        }

        for i in 0..self.layers.len() {
            self.layers[i].update_weights(learning_rate);
        }

        if self.weight_decay.is_some() {
            self.optimise();
        }
    }

    fn optimise(&mut self) {
        for i in 0..self.layers.len() {
            let weights = &self.layers[i].weights().clone();
            self.layers[i].set_weights(&matrix_math::matrix_subtract_flat(weights, &matrix_math::matrix_scalar_multiply(weights, self.weight_decay.unwrap() as f32)).unwrap());
        }    
    }

    fn calculate_deltas(&self, layer: &Layer, next_layer: &Layer, next_delta: &FlatMatrix) -> FlatMatrix {
        // delta_prev = delta*W hadamard product with the derivative of the activation function
        // Hadamard Product - multiply every element in the matrix by the operand

        let multiply = matrix_math::matrix_multiply_flat(next_delta, next_layer.weights(), false).unwrap();
        let derivatives = layer.activation_derivative();

        let out = matrix_math::matrix_hadamard_product(&multiply, &derivatives).unwrap();

        return out;
    }

    fn calculate_loss_derivative(&self, outputs: &FlatMatrix, expected_outputs: &FlatMatrix) -> Result<FlatMatrix, String> {
        if outputs.rows != expected_outputs.rows || outputs.cols() != expected_outputs.cols() {
            return Err("Matrix orders differ".to_string());
        }

        let out;

        let divisor: f32 = 1.0 / outputs.cols() as f32;

        if self.softmax_enabled {
            out = outputs.mat.iter().zip(&expected_outputs.mat).map(|(a, b)| (a - b) * divisor).collect();
        } else {
            out = outputs.mat.iter().zip(&expected_outputs.mat).map(|(a, b)| 2.0*(a - b)).collect();
        }

        return Ok(FlatMatrix { mat: out, rows: outputs.rows });
    }

    fn calculate_loss(&self, outputs: &FlatMatrix, expected_outputs: &FlatMatrix) -> Result<FlatMatrix, String> {
        if outputs.rows != expected_outputs.rows || outputs.cols() != expected_outputs.cols() {
            return Err("Matrix orders differ".to_string());
        }

        let out: Vec<f32>;
        
        if self.softmax_enabled {
            out = outputs.mat.iter().zip(&expected_outputs.mat).map(|(a, b)| if *b == 0.0 {0.0} else {-b*f32::ln(*a)}).collect();
        } else {
            out = outputs.mat.iter().zip(&expected_outputs.mat).map(|(a, b)| f32::powi(a - b, 2)).collect();
        }

        return Ok(FlatMatrix { mat: out, rows: outputs.rows });
    }
}