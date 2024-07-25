use ferrograd::{
    engine::{Activation, Value},
    loss::{softmax, CrossEntropyLoss},
    metrics::BinaryAccuracy,
    nn::{
        optim::{l2_regularization, Adam},
        MultiLayerPerceptron,
    },
};
use rust_mnist::{print_image, Mnist};

fn main() {
    // Loading training data
    let mnist = Mnist::new("data/mnist/");
    let xtrain: Vec<Vec<Value>> = mnist.train_data[..200] // training with only 200 samples.
        .iter()
        .map(|img| {
            img.iter()
                .map(|pix| Value::new(*pix as f64 / 255.0))
                .collect()
        })
        .collect();
    let ytrain: Vec<Vec<Value>> = mnist.train_labels[..200]
        .iter()
        .map(|label| one_hot(*label))
        .collect();

    // MLP
    let model =
        MultiLayerPerceptron::new(764, vec![32, 32, 10], Activation::ReLU);
    println!("Model: {:#?}\n", model);

    // Optimizer, loss criterion and accuracy metric
    let mut optim = Adam::new(model.parameters(), 0.1, 0.9, 0.999, 1e-4);
    let loss = CrossEntropyLoss::new();
    let accuracy = BinaryAccuracy::new(0.5);

    println!(
        "Optimiser: {:#?}\n\nCriterion: {:#?}\n\nMetric: {:#?}\n",
        optim, loss, accuracy
    );

    // Training
    (0..100).for_each(|k| {
        let ypred = softmax(&model.forward(&xtrain));

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
    });

    println!();

    // Loading training data
    let xtest: Vec<Vec<Value>> = mnist.test_data[..10]
        .iter()
        .map(|img| {
            img.iter()
                .map(|pix| Value::new(*pix as f64 / 255.0))
                .collect()
        })
        .collect();

    // Making predictions
    let ypred = softmax(&model.forward(&xtest));
    let mut correct = 0;
    let total = 10;

    for i in 0..total {
        let argmax = ypred[i]
            .iter()
            .enumerate()
            .max_by_key(|(_, v)| *v)
            .map(|(ind, _)| ind)
            .expect("Error  in prediction");

        let img = &mnist.test_data[i];
        let label = mnist.test_labels[i];

        if label as usize == argmax {
            correct += 1
        }

        print_image(img, label);
        println!("Prediction: {}\n", argmax);
    }

    println!("Correct predictions: {}/{}", correct, total);
}

fn one_hot(digit: u8) -> Vec<Value> {
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
    vec.iter().map(|v| Value::new(*v as f64)).collect()
}
