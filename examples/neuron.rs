use micrograd::{
    engine::{Activation, Value},
    nn::Neuron,
};

fn main() {
    let n = Neuron::new(5, Some(Activation::ReLU)).with_names();
    let x = vec![
        Value::new(1.0),
        Value::new(-1.0),
        Value::new(1.5),
        Value::new(-4.0),
        Value::new(1.75),
        Value::new(-2.0),
    ];

    let y = n.forward(&x);
    println!("{}", y.tree());
}
