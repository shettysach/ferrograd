use ferrograd::{
    engine::{Activation, Value},
    loss::CrossEntropyLoss,
    metrics::BinaryAccuracy,
    nn::{
        optim::{l2_regularization, Adam},
        softmax, MultiLayerPerceptron,
    },
};

fn main() {
    // Loading training data
    let batch_size = 100;
    let mnist = rust_mnist::Mnist::new("data/mnist/");

    // MLP
    let model =
        MultiLayerPerceptron::new(784, vec![64, 32, 10], Activation::LeakyReLU);
    println!("Model: {:#?}\n", model);

    // Optimizer, loss criterion and accuracy metric
    let mut optim = Adam::new(model.parameters(), 0.1, 0.9, 0.999, 1e-8);
    let loss = CrossEntropyLoss::new();
    let accuracy = BinaryAccuracy::new(0.5);

    println!(
        "Optimiser: {:#?}\n\nCriterion: {:#?}\n\nMetric: {:#?}\n",
        optim, loss, accuracy
    );

    let model_path = "model/mod_64x32";

    // Training
    (0..200).for_each(|k| {
        let b = k % 10;
        let start = b * batch_size;
        let end = (b + 1) * batch_size;

        let xtrain = images_to_features(&mnist.train_data[start..end]);
        let ytrain = one_hot_encode(&mnist.train_labels[start..end]);

        let ypred = model.forward(&xtrain);
        let ypred = softmax(&ypred);

        let data_loss = loss.loss(&ypred, &ytrain);
        let reg_loss = l2_regularization(0.0001, model.parameters());
        let total_loss = data_loss + reg_loss;

        optim.zero_grad();
        total_loss.backward();
        optim.step();

        let acc = accuracy.compute(&ypred, &ytrain);

        println!(
            "step {} - loss {:.3}, accuracy {:.2}%",
            k,
            total_loss.borrow().data,
            acc * 100.0
        );

        // Save at every 10th step
        if b == 9 {
            match model.save(model_path) {
                Ok(_) => {
                    println!("\n> Model saved successfully at {model_path}\n")
                }
                Err(err) => eprintln!("{}", err),
            };
        }
    });

    // Loading training data
    let test_samples = 100;
    let xtest: Vec<Vec<Value>> = mnist.test_data[..test_samples]
        .iter()
        .map(|img| {
            img.iter()
                .map(|pix| Value::new(*pix as f64 / 255.0))
                .collect()
        })
        .collect();

    // Making predictions
    println!("Testing");
    let ypred = softmax(&model.forward(&xtest));

    let correct = ypred
        .iter()
        .enumerate()
        .take(test_samples)
        .filter(|(i, ypred_i)| {
            let (argmax, _) = ypred_i
                .iter()
                .enumerate()
                .max_by_key(|(_, v)| *v)
                .map(|(ind, v)| (ind, v.borrow().data))
                .expect("Error  in prediction");

            let label = mnist.test_labels[*i];
            label as usize == argmax
        })
        .count();

    println!("Correct predictions: {}/{}", correct, test_samples);
}

// --- Data transformation ---

fn images_to_features(imgvec: &[[u8; 784]]) -> Vec<Vec<Value>> {
    imgvec
        .iter()
        .map(|img| {
            img.iter()
                .map(|pix| Value::new(*pix as f64 / 255.0))
                .collect()
        })
        .collect()
}

fn one_hot_encode(labels: &[u8]) -> Vec<Vec<Value>> {
    labels
        .iter()
        .map(|digit| {
            let vec = match digit {
                0 => vec![1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                1 => vec![0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                2 => vec![0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                3 => vec![0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                4 => vec![0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                5 => vec![0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                6 => vec![0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                7 => vec![0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                8 => vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                9 => vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                _ => panic!("Invalid digit"),
            };
            Value::from_1d(&vec)
        })
        .collect()
}
