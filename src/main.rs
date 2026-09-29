pub mod neural_network;
pub mod matrix_math;

use neural_network::Network;

use rand::seq::SliceRandom;

use matrix_math::FlatMatrix;

type Sample = (FlatMatrix, FlatMatrix);

fn main() {
    let mut net = Network::new();

    let dataset = create_mod_dataset(97, 0.5);

    let train_set = dataset.0;
    let test_set = dataset.1;

    for i in 0..10 {
        train_mod_addition(&mut net, Some(train_set.clone()));
        evaluate_network(&mut net, &train_set, &test_set);
    }

}

fn train_mod_addition(network: &mut Network, training_data: Option<Vec<Sample>>) {

    let mut train_set = if let Some(data) = training_data {
        data
    } else {
        create_mod_dataset(97, 0.8).0
    };

    let mut rng = rand::rng();

    let learning_rate = 0.1;
    let epochs = 100; // How many times the algorithm goes over the training data

    for _epoch in 0..epochs {
        train_set.shuffle(&mut rng);

        for (inputs, targets) in train_set.iter() {
            network.back_prop(inputs, targets, learning_rate);
        }
    }


    // let mut self_test = vec![0.0; 194];

    // let x = 22;
    // let y = 81;

    // self_test[x] = 1.0;
    // self_test[97 + y] = 1.0;

    // println!("{:?}", network.forward_pass(FlatMatrix { mat: self_test, rows: 1 }));
}


fn create_mod_dataset(p: usize, train_ratio: f64) -> (Vec<Sample>, Vec<Sample>) {
    let mut dataset: Vec<Sample> = Vec::with_capacity(p * p);

    let mut batch_in = FlatMatrix {mat: vec![], rows: 0};
    let mut batch_target = FlatMatrix {mat: vec![], rows: 0};


    for x in 0..p {
        for y in 0..p {
            let target_idx = (x + y) % p;

            // 1. Create One-Hot Input Vector (length: 2 * p)
            let mut input = vec![0.0; 2 * p];
            input[x] = 1.0;          // One-hot slot for X
            input[p + y] = 1.0;      // One-hot slot for Y

            // 2. Create One-Hot Target Vector (length: p)
            let mut target = vec![0.0; p];
            target[target_idx] = 1.0;

            batch_in.mat.extend(input.clone());
            batch_in.rows = batch_in.rows + 1;
            batch_target.mat.extend(target.clone());
            batch_target.rows = batch_target.rows + 1;


            if &batch_in.rows % 32 as usize == 0|| (x == p-1 && y == p-1) {
                dataset.push((batch_in.clone(), batch_target.clone()));
                batch_in = FlatMatrix {mat: vec![], rows: 0};
                batch_target = FlatMatrix {mat: vec![], rows: 0};
            }
        }
    }

    // 3. Shuffle dataset randomly
    let mut rng = rand::rng();
    dataset.shuffle(&mut rng);

    // 4. Split into train and test sets
    let train_size = ((dataset.len() as f64) * train_ratio) as usize;
    let test_set = dataset.split_off(train_size); // Retains 0..train_size in `dataset`, returns remainder
    let train_set = dataset;

    (train_set, test_set)
}


fn argmax(slice: &[f32]) -> usize {
    slice
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("NaN encountered in network output"))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

fn evaluate_network(network: &mut Network, train_set: &[Sample], test_set: &[Sample]) {
    let mut correct = 0;

    let mut total = 0;

    for (inputs, targets) in train_set {
        // 1. Run forward pass on the 194-element input
        let predictions = network.forward_pass(inputs.clone()); // Ensure infer() returns Vec<f64> or &Vec<f64>

        let rows = predictions.rows;
        let cols = predictions.cols();

        // 2. Find the predicted class (highest activation) and target class
        for i in 0..rows {
            let predicted_class = argmax(&predictions.mat[cols*i..cols*(i+1)]);
            let target_class = argmax(&targets.mat[cols*i..cols*(i+1)]);

            // 3. Count matching predictions
            if predicted_class == target_class {
                correct += 1;
            }

            total += 1;
        }
    }

    let train_accuracy = (correct as f64 / total as f64) * 100.0;

    println!("Train Set Accuracy: {:.2}% ({}/{})", train_accuracy, correct, total);

    let mut correct = 0;
    let mut total = 0;

    for (inputs, targets) in test_set {
        // 1. Run forward pass on the 194-element input
        let predictions = network.forward_pass(inputs.clone()); // Ensure forward_pass() returns Vec<f64> or &Vec<f64>

        let rows = predictions.rows;
        let cols = predictions.cols();

        // 2. Find the predicted class (highest activation) and target class
        for i in 0..rows {
            let predicted_class = argmax(&predictions.mat[cols*i..cols*(i+1)]);
            let target_class = argmax(&targets.mat[cols*i..cols*(i+1)]);

            // 3. Count matching predictions
            if predicted_class == target_class {
                correct += 1;
            }

            total += 1;
        }
    }

    let test_accuracy = (correct as f64 / total as f64) * 100.0;

    println!("Test Set Accuracy: {:.2}% ({}/{})", test_accuracy, correct, total);
}