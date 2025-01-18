use ferrograd::{
    engine::{ActvFn, Value},
    nn::MultiLayerPerceptron,
};

fn main() {
    let mlp = MultiLayerPerceptron::new(2, vec![2, 1, 1], ActvFn::Sigmoid);
    let y = mlp.forward(&Value::new_2d(&[&[1.0, 2.0]]));
    y[0][0].backward();
    println!("{}", y[0][0].tree());
}
