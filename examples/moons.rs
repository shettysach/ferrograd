use ferrograd::{
    engine::{ActvFn, Value},
    loss::HingeLoss,
    metrics::BinaryAccuracy,
    nn::{
        optim::{l2_regularization, SGD},
        MultiLayerPerceptron,
    },
    utils::read_csv,
};

fn main() {
    let (xs, ys) = read_csv("data/moons_data.csv", &[0, 1], &[2], 1);

    let model = MultiLayerPerceptron::new(2, vec![16, 16, 1], ActvFn::ReLU);
    println!("{}", model);
    println!("Number of parameters: {}\n", model.parameters().len());

    let mut optim = SGD::new(model.parameters(), 0.1, 0.9);
    let loss = HingeLoss::new();
    let accuracy = BinaryAccuracy::new(0.0);

    println!(
        "Optimiser: {:#?}\n\nCriterion: {:#?}\n\nMetric: {:#?}\n",
        optim, loss, accuracy
    );

    for k in 0..100 {
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
    }

    println!();
    print_grid(&model, 15);
}

fn print_grid(model: &MultiLayerPerceptron, bound: i32) {
    println!("\nASCII contour graph - \n■ > 0.0  \n□ <= 0.0 ");
    for y in -bound..bound {
        for x in -bound..bound {
            let k = &model.forward(&[vec![
                Value::new(x as f64 / bound as f64 * 2.0),
                Value::new(-y as f64 / bound as f64 * 2.0),
            ]])[0][0];

            if k.borrow().data > 0.0 {
                print!("■ ");
            } else {
                print!("□ ");
            }
        }
        println!();
    }
}
