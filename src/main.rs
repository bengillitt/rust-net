pub mod neural_network;
pub mod matrix_math;

use neural_network::Network;

use crate::matrix_math::transpose_mat;

fn main() {
    let mat_1 = matrix_math::FlatMatrix{mat: vec![1.0, 1.0, 0.0, 0.0, 1.0, 1.0], rows: 2};
    let mat_2 = matrix_math::FlatMatrix{mat: vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0], rows: 2};
    
    println!("{:?}", mat_1);

    println!("{:?}", mat_2);


    println!("{:?}", matrix_math::matrix_multiply_flat(&mat_1, &mat_2, true));
}
