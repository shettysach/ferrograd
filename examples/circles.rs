use ferrograd::{
    engine::{Activation, Value},
    loss::MeanSquareErrorLoss,
    metrics::BinaryAccuracy,
    nn::{
        optim::{l2_regularization, Adam},
        MultiLayerPerceptron,
    },
};

fn main() {
    let (xs, ys) = load_data("data/circles_data.csv", 1);

    let model = MultiLayerPerceptron::new(2, vec![24, 24, 1], Activation::ReLU);
    println!("Model - \n{}", model);
    println!("Number of parameters = {}\n", model.parameters().len());

    let mut optim = Adam::new(model.parameters(), 0.01, 0.9, 0.999, 1e-7);
    let loss = MeanSquareErrorLoss::new();
    let accuracy = BinaryAccuracy::new(0.5);

    (0..100).for_each(|k| {
        let ypred: Vec<Vec<Value>> = model.forward(&xs);

        let data_loss = loss.loss(&ypred, &ys).pow(0.5); // RMSE
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

    print_grid(&model, 15);
}

// -- Grid --

fn print_grid(model: &MultiLayerPerceptron, bound: i32) {
    println!("\nASCII contour graph - \n■ > 0.0  \n□ <= 0.0 ");
    let grid: Vec<Vec<&str>> = (-bound..bound)
        .map(|y| {
            (-bound..bound)
                .map(|x| {
                    let k = &model.forward(&vec![
                        vec![Value::new(x as f64 / bound as f64 * 2.0)],
                        vec![Value::new(-y as f64 / bound as f64 * 2.0)],
                    ])[0][0];

                    if k.borrow().data > 0.0 {
                        "■"
                    } else {
                        "□"
                    }
                })
                .collect()
        })
        .collect();

    println!();
    grid.iter().for_each(|row| println!("{}", row.join(" ")));
}

// --- Dataloader ---

use std::fs::File;
use std::io::{BufRead, BufReader};

fn load_data(
    filepath: &str,
    skip_rows: usize,
) -> (Vec<Vec<Value>>, Vec<Vec<Value>>) {
    let file = File::open(filepath).unwrap();
    let reader = BufReader::new(file);

    let mut x0s = Vec::new();
    let mut x1s = Vec::new();
    let mut y0s = Vec::new();

    reader.lines().skip(skip_rows).for_each(|line| {
        let line = line.unwrap();
        let fields: Vec<&str> = line.split(',').collect();

        let x0 = fields[0].parse::<f64>().unwrap();
        let x1 = fields[1].parse::<f64>().unwrap();
        let y0 = fields[2].parse::<f64>().unwrap();

        x0s.push(Value::new(x0));
        x1s.push(Value::new(x1));
        y0s.push(Value::new(y0));
    });

    let xs = vec![x0s, x1s];
    let ys = vec![y0s];

    (xs, ys)
}
