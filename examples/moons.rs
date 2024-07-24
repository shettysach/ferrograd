use ferrograd::{
    engine::{Activation, Value},
    loss::HingeLoss,
    metrics::BinaryAccuracy,
    nn::{
        optim::{l2_regularization, SGD},
        MultiLayerPerceptron,
    },
    utils::read_csv,
};

fn main() {
    let (xs, ys) = read_csv("data/moons_data.csv", 2, 1, 1);

    let model = MultiLayerPerceptron::new(2, vec![16, 16, 1], Activation::ReLU);
    println!("Model - \n{}", model);
    println!("Number of parameters = {}\n", model.parameters().len());

    let mut optim = SGD::new(model.parameters(), 0.1, 0.9);
    let loss = HingeLoss::new();
    let accuracy = BinaryAccuracy::new(0.0);

    (0..100).for_each(|k| {
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
