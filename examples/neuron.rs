use ferrograd::{
    engine::{Activation, Value},
    nn::Neuron,
};

fn main() {
    let n = Neuron::new(2, Some(Activation::ReLU));
    let n = n.name_params();

    let x = Value::new_2d(&[vec![2.0, 1.0]]);
    let x = n.name_inputs(x);

    println!("{}\n", n);

    let y = &n.forward(&x)[0];
    println!("Forward pass: \n{}", y.tree());

    y.backward();
    println!("Backward pass: \n{}", y.tree());
}
