pub mod neural_network;
pub mod matrix_math;

use neural_network::Network;

fn main() {
    let mut net = Network::new();

    println!("Forward Prop Test: {:?}", net.forward_pass(matrix_math::FlatMatrix { mat: vec![1.0, 1.0, 0.0, 0.0], rows: (2) }))
}
