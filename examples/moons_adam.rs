use micrograd::{
    engine::{Activation, Value},
    nn::MultiLayerPerceptron,
    optim::adam::Adam,
};

fn main() {
    let model =
        MultiLayerPerceptron::new(2, vec![16, 16, 1], Activation::LeakyReLU);
    let p = model.parameters().len();
    println!("Number of parameters = {}\n", p);

    let (xs, ys) = load_moons_data();
    let mut optim = Adam::new(model.parameters(), 0.1, 0.9, 0.999, 0.00000001);

    for k in 0..100 {
        let (total_loss, acc) = loss(&model, &xs, &ys);

        optim.zero_grad();
        total_loss.backward();

        optim.step();

        println!(
            "step {} - loss {:.3}, accuracy {:.2}%",
            k,
            total_loss.borrow().data,
            acc * 100.0
        );
    }

    // ASCII contour plot
    let bound = 15;
    let grid: Vec<Vec<String>> = (-bound..bound)
        .map(|y| {
            (-bound..bound)
                .map(|x| {
                    let k = &model.forward(vec![
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

fn loss(
    model: &MultiLayerPerceptron,
    xs: &[Vec<f64>],
    ys: &[f64],
) -> (Value, f64) {
    let inputs: Vec<Vec<Value>> = xs
        .iter()
        .map(|xrow| vec![Value::new(xrow[0]), Value::new(xrow[1])])
        .collect();

    // forward the model to get scores
    let scores: Vec<Value> = inputs
        .iter()
        .map(|xrow| model.forward(xrow.clone())[0].clone())
        .collect();

    // svm "max-margin" loss
    let losses: Vec<Value> = ys
        .iter()
        .zip(&scores)
        .map(|(yi, scorei)| (1.0 + -yi * scorei).relu())
        .collect();
    let n: f64 = (&losses).len() as f64;
    let data_loss: Value = losses.into_iter().sum::<Value>() / n;

    // L2 regularization
    let alpha: f64 = 0.0001;
    let reg_loss: Value = alpha
        * model
            .parameters()
            .iter()
            .map(|p| p * p)
            .into_iter()
            .sum::<Value>();
    let total_loss = data_loss + reg_loss;

    let accuracies: Vec<bool> = ys
        .iter()
        .zip(scores.iter())
        .map(|(yi, scorei)| (*yi > 0.0) == (scorei.borrow().data > 0.0))
        .collect();
    let accuracy = accuracies.iter().filter(|&a| *a).count() as f64 / n;

    (total_loss, accuracy)
}

// --- Dataloader ---
// https://github.com/Mathemmagician/rustygrad/blob/dev/src/utils.rs

use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};

pub struct DataPoint {
    pub x: f64,
    pub y: f64,
    pub label: f64,
}

fn read_moons_csv(filename: &str) -> Result<Vec<DataPoint>, Box<dyn Error>> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);

    let mut data_points = vec![];

    for (index, line) in reader.lines().enumerate() {
        let line = line?;

        if index == 0 {
            // Skip the header row
            continue;
        }

        let fields: Vec<&str> = line.split(',').collect();

        let x = fields[0].parse::<f64>()?;
        let y = fields[1].parse::<f64>()?;
        let label = fields[2].parse::<f64>()?;

        let data_point = DataPoint { x, y, label };
        data_points.push(data_point);
    }

    Ok(data_points)
}

fn load_moons_data() -> (Vec<Vec<f64>>, Vec<f64>) {
    let data_points =
        read_moons_csv("data/moons_data.csv").expect("Error reading moons.csv");

    let (xs, ys) = data_points
        .iter()
        .map(|data_point| {
            let x_vec = vec![data_point.x, data_point.y];
            (x_vec, data_point.label)
        })
        .collect();

    (xs, ys)
}
