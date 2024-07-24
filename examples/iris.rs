use ferrograd::{
    engine::{Activation, Value},
    loss::BinaryCrossEntropyLoss,
    metrics::BinaryAccuracy,
    nn::{
        optim::{l2_regularization, Adam},
        MultiLayerPerceptron,
    },
};

fn main() {
    let (xs, ys) = read_iris_csv();

    let model = MultiLayerPerceptron::new(4, vec![16, 16, 3], Activation::ReLU);
    println!("Model - \n{}", model);
    println!("Number of parameters = {}\n", model.parameters().len());

    let mut optim = Adam::new(model.parameters(), 0.1, 0.9, 0.999, 0.00000001);
    let loss = BinaryCrossEntropyLoss::new();
    let accuracy = BinaryAccuracy::new(0.0);

    (0..150).for_each(|k| {
        let ypred: Vec<Vec<Value>> = model.forward(&xs);

        let data_loss = loss.loss(&ypred, &ys);
        let reg_loss = l2_regularization(0.0001, model.parameters());
        let total_loss = data_loss + reg_loss;

        optim.zero_grad();
        total_loss.backward();
        optim.step();

        let acc = accuracy.compute(&ypred, &ys);

        println!(
            "step {} - loss {:.3}, accuracy {:.2}%",
            k,
            total_loss.borrow().data,
            acc * 100.0
        );
    });
}

// --- Dataloader ---
// https://github.com/Mathemmagician/rustygrad/blob/dev/src/utils.rs

use std::fs::File;
use std::io::{BufRead, BufReader};

fn read_iris_csv() -> (Vec<Vec<Value>>, Vec<Vec<Value>>) {
    let file = File::open("data/iris_data.csv").unwrap();
    let reader = BufReader::new(file);

    let (xs, ys) = reader
        .lines()
        .map(|line| {
            let line = line.unwrap();
            let fields: Vec<&str> = line.split(',').collect();

            let x_vec = (0..4)
                .map(|i| Value::new(fields[i].parse::<f64>().unwrap()))
                .collect();

            let y_vec = match fields[4] {
                "Iris-setosa" => {
                    vec![Value::new(1.0), Value::new(0.0), Value::new(0.0)]
                }
                "Iris-versicolor" => {
                    vec![Value::new(0.0), Value::new(1.0), Value::new(0.0)]
                }
                "Iris-virginica" => {
                    vec![Value::new(0.0), Value::new(0.0), Value::new(1.0)]
                }
                _ => panic!("Unknown species"),
            };

            (x_vec, y_vec)
        })
        .collect();

    (xs, ys)
}
