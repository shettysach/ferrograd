use ferrograd::{
    engine::{Activation, Value},
    loss::softmax,
    nn::MultiLayerPerceptron,
    utils::print_mnist_image,
};
use rand::Rng;

fn main() {
    // MLP
    let mnist = rust_mnist::Mnist::new("data/mnist/");

    let model =
        MultiLayerPerceptron::new(784, vec![64, 32, 10], Activation::LeakyReLU);
    println!("{}\n", model);

    let model_path = "model/mod_64x32";
    match model.load(model_path) {
        Ok(_) => println!("> Model loaded successfully from {model_path}\n"),
        Err(err) => panic!("{}", err),
    };

    // Loading test data
    let test_samples = 100;
    let offset = rand::thread_rng().gen_range(0..9_900);

    let xtest: Vec<Vec<Value>> = mnist.test_data[offset..offset + test_samples]
        .iter()
        .map(|img| {
            img.iter()
                .map(|pix| Value::new(*pix as f64 / 255.0))
                .collect()
        })
        .collect();

    // Making predictions
    let mut correct = 0;
    xtest.iter().enumerate().for_each(|(i, x)| {
        let ypred = softmax(&vec![model.forw(&x)]);
        let (argmax, prob) = ypred[0]
            .iter()
            .enumerate()
            .max_by_key(|(_, v)| *v)
            .map(|(ind, v)| (ind, v.borrow().data))
            .expect("Error  in prediction");

        let img = &mnist.test_data[offset + i];
        let label = mnist.test_labels[offset + i];

        let pred = label as usize == argmax;

        if label as usize == argmax {
            correct += 1
        }

        print_mnist_image(img);
        println!("ytrue: {}", label);
        println!("ypred: {argmax}");
        println!("prob: {prob:.3}");
        println!("pred: {pred}\n");
    });

    println!("Correct predictions: {}/{}", correct, test_samples);
}