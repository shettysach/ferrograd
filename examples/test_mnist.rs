use ferrograd::{
    engine::{Activation, Value},
    nn::{softmax, MultiLayerPerceptron},
};
use rand::Rng;

fn main() {
    // MLP
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

    let mnist = rust_mnist::Mnist::new("data/mnist/");
    let xtest: Vec<Vec<Value>> =
        images_to_features(&mnist.test_data[offset..offset + test_samples]);

    // Making predictions
    let correct = xtest
        .iter()
        .enumerate()
        .filter(|(i, x)| {
            let ypred = vec![model.forw(x)];
            let ypred = softmax(&ypred);

            let (argmax, prob) = ypred[0]
                .iter()
                .enumerate()
                .max_by_key(|(_, v)| *v)
                .map(|(ind, v)| (ind, v.borrow().data))
                .expect("Error  in prediction");

            let img = &mnist.test_data[offset + i];
            let label = mnist.test_labels[offset + i];
            let pred = label as usize == argmax;

            print_mnist_image(img);
            println!("ytrue: {}", label);
            println!("ypred: {argmax}");
            println!("prob: {prob:.3}");
            println!("pred: {pred}\n");

            pred
        })
        .count();

    println!("Correct predictions: {}/{}", correct, test_samples);
}

// --- Helper functions ---

fn images_to_features(imgvec: &[[u8; 784]]) -> Vec<Vec<Value>> {
    imgvec
        .iter()
        .map(|img| {
            img.iter()
                .map(|pix| Value::new(*pix as f64 / 255.0))
                .collect()
        })
        .collect()
}

fn print_mnist_image(image: &[u8; 28 * 28]) {
    (0..28).for_each(|row| {
        (0..28).for_each(|col| {
            if image[row * 28 + col] == 0 {
                print!("□ ");
            } else {
                print!("■ ");
            }
        });
        println!();
    });
}
