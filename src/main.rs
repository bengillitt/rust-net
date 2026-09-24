pub mod neural_network;
pub mod matrix_math;

use neural_network::Network;

fn main() {
    let mat = matrix_math::FlatMatrix{mat: vec![1.0, 2.0, 3.0, 4.0], rows: 2};

    println!("{:?}", matrix_math::matrix_multiply_flat(&mat.clone(), &mat.clone(), false));
}
