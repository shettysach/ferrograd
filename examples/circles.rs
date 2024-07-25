use ferrograd::{
    engine::{Activation, Value},
    loss::BinaryCrossEntropyLoss,
    metrics::BinaryAccuracy,
    nn::{
        optim::{l2_regularization, Adam},
        MultiLayerPerceptron,
    },
    utils::read_csv,
};

fn main() {
    let (xs, ys) = read_csv("data/circles_data.csv", &[0, 1], &[2], 1);

    let model = MultiLayerPerceptron::new(2, vec![16, 16, 1], Activation::ReLU);
    println!("Model: {:#?}\n", model);

    let mut optim = Adam::new(model.parameters(), 0.1, 0.9, 0.999, 1e-4);
    let loss = BinaryCrossEntropyLoss::new();
    let accuracy = BinaryAccuracy::new(0.5);

    println!(
        "Optimiser: {:#?}\n\nCriterion: {:#?}\n\nMetric: {:#?}\n",
        optim, loss, accuracy
    );

    (0..100).for_each(|k| {
        let ypred = model.forward(&xs);

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

    print_grid(&model, 15);
}

// -- Grid --

fn print_grid(model: &MultiLayerPerceptron, bound: i32) {
    println!("\nASCII contour graph - \n■ > 0.5  \n□ <= 0.5 ");
    let grid: Vec<Vec<&str>> = (-bound..bound)
        .map(|y| {
            (-bound..bound)
                .map(|x| {
                    let k = &model.forward(&vec![vec![
                        Value::new(x as f64 / bound as f64 * 2.0),
                        Value::new(-y as f64 / bound as f64 * 2.0),
                    ]])[0][0];

                    if k.borrow().data > 0.5 {
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
