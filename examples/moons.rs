use micrograd::{
    engine::{Activation, Value},
    loss::HingeEmbeddingLoss,
    metrics::BinaryAccuracy,
    nn::{
        optim::{l2_regularization, SGD},
        MultiLayerPerceptron,
    },
};

fn main() {
    let model = MultiLayerPerceptron::new(2, vec![16, 16, 1], Activation::ReLU);

    println!("Model - \n{}", model);
    println!("Number of parameters = {}\n", model.parameters().len());

    let (xs, ys) = load_data();
    let mut optim = SGD::new(model.parameters(), 0.1, 0.9);
    let loss = HingeEmbeddingLoss::new(1.0);
    let accuracy = BinaryAccuracy::new(0.0);

    (0..100).for_each(|k| {
        let ypred: Vec<Vec<Value>> = model.forward_batch(&xs);

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

    println!("\nASCII contour graph - \n■ > 0.0  \n□ <= 0.0 ");
    let bound = 15;
    let grid: Vec<Vec<String>> = (-bound..bound)
        .map(|y| {
            (-bound..bound)
                .map(|x| {
                    let k = &model.forward(&vec![
                        Value::new(x as f64 / bound as f64 * 2.0),
                        Value::new(-y as f64 / bound as f64 * 2.0),
                    ])[0];

                    if k.borrow().data > 0.0 {
                        String::from("■")
                    } else {
                        String::from("□")
                    }
                })
                .collect()
        })
        .collect();

    println!();
    grid.iter().for_each(|row| println!("{}", row.join(" ")));
}

// --- Dataloader ---
// https://github.com/Mathemmagician/rustygrad/blob/dev/src/utils.rs

use std::fs::File;
use std::io::{BufRead, BufReader};

pub struct DataPoint {
    pub x: f64,
    pub y: f64,
    pub label: f64,
}

fn read_csv(filename: &str) -> Vec<DataPoint> {
    let file = File::open(filename).unwrap();
    let reader = BufReader::new(file);

    reader
        .lines()
        .skip(1)
        .map(|line| {
            let line = line.unwrap();
            let fields: Vec<&str> = line.split(',').collect();

            let x = fields[0].parse::<f64>().unwrap();
            let y = fields[1].parse::<f64>().unwrap();
            let label = fields[2].parse::<f64>().unwrap();

            DataPoint { x, y, label }
        })
        .collect()
}

fn load_data() -> (Vec<Vec<Value>>, Vec<Vec<Value>>) {
    let data_points = read_csv("data/moons_data.csv");

    let (xs, ys) = data_points
        .iter()
        .map(|data_point| {
            let x_vec =
                vec![Value::new(data_point.x), Value::new(data_point.y)];
            let y_vec = vec![Value::new(data_point.label)];

            (x_vec, y_vec)
        })
        .collect();

    (xs, ys)
}
