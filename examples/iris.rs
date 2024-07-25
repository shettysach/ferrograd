use ferrograd::{
    engine::{Activation, Value},
    loss::{softmax, CrossEntropyLoss},
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
    let loss = CrossEntropyLoss::new();
    let accuracy = BinaryAccuracy::new(0.5);

    println!(
        "Optimiser: {:#?}\n\nCriterion: {:#?}\n\nMetric: {:#?}\n",
        optim, loss, accuracy
    );

    (0..100).for_each(|k| {
        let ypred = softmax(&model.forward(&xs));

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

    let samples = vec![
        vec![5.1, 3.5, 1.4, 0.2],
        vec![7.2, 2.7, 6.0, 2.0],
        vec![5.8, 2.7, 3.9, 1.2],
    ];
    let x = samples
        .iter()
        .map(|vec| vec.iter().map(|f| Value::new(*f)).collect())
        .collect();

    let preds = model.forward(&x);

    println!();
    for (sample, pred) in samples.iter().zip(preds) {
        let argmax = pred
            .iter()
            .enumerate()
            .max_by_key(|(_, v)| *v)
            .map(|(ind, _)| ind);

        let res = match argmax {
            Some(0) => Ok("Iris-setosa"),
            Some(1) => Ok("Iris-versicolor"),
            Some(2) => Ok("Iris-virginica"),
            _ => Err("Error predicting value"),
        };

        print!("X: {:?} => ", sample);
        println!("ypred: {:?}", res);
    }
}

// --- Dataloader ---

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
