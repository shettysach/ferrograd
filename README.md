### ferrograd

- A small autograd engine, inspired from [karpathy/micrograd](https://github.com/karpathy/micrograd), with more features, such as additional activation functions, optimizers, loss criterions, accuracy metrics, and the ability to save and load model weights after training.
- Created for learning purposes. See `notes` directory. 
- Capable of creating neurons, dense layers and multilayer perceptrons, for non-linear classification tasks.

> NOTE: 
> - To run the MNIST example, download the [data](https://yann.lecun.com/exdb/mnist/), gzip extract all 4 files, and move them to the directory `/data/mnist/`.  
> - Not optimized for performance and uses scalar values (`Value`) and operations. `Vec<Value>` and `Vec<Vec<Value>>` are used in place of tensors.
> - Run examples with the `release` flag (`cargo run --release --example <example>`) for more optimized performance.

```zsh
cargo add --git https://github.com/shettysach/ferrograd.git ferrograd
```

#### Examples

##### Readme example from karpathy/micrograd

```rust
use ferrograd::engine::Value;

fn main() {
    let a = Value::new(-4.).with_name("a");
    let b = Value::new(2.).with_name("b");

    let mut c = (&a + &b).with_name("c");
    let mut d = (&a * &b + &b.pow(3.)).with_name("d");

    c += &c + 1.;
    c += 1. + &c + (-&a);
    d += &d * 2. + (&b + &a).relu();
    d += 3. * &d + (&b - &a).relu();

    let e = (&c - &d).with_name("e");
    let f = e.pow(2.).with_name("f");

    let mut g = (&f / 2.).with_name("g");
    g += 10. / &f;

    g.backward();
    println!("{}", g.tree());

    println!("g.data = {:.4}", g.borrow().data);
    println!("a.grad = {:.4}", a.borrow().grad);
    println!("b.grad = {:.4}", b.borrow().grad);
}
```

```console
cargo run --example readme
```

```
# Printing of g.tree()

g.data = 24.7041
a.grad = 138.8338
b.grad = 645.5773
```

---

##### Neuron

```rust
use ferrograd::{
    engine::{Activation, Value},
    nn::Neuron,
};

fn main() {
    let n = Neuron::new(2, Some(Activation::ReLU));
    let n = n.name_params();

    let x = Value::new_2d(&vec![vec![2.0, 1.0]]);
    let x = n.name_inputs(x);

    println!("{}\n", n);

    let y = &n.forward(&x)[0];
    println!("Forward pass:\n{}", y.tree());

    y.backward();
    println!("Backward pass:\n{}", y.tree());
}
```

```console
cargo run --example neuron
```

```
ReLU(2)

Forward pass:
ReLU data = 1.800, grad = 0.000
└── + data = 1.800, grad = 0.000
    ├── + data = 1.800, grad = 0.000
    │   ├── * data = 1.911, grad = 0.000
    │   │   ├── data = 0.955, grad = 0.000 ← weight[0]
    │   │   └── data = 2.000, grad = 0.000 ← x[0][0]
    │   └── * data = -0.111, grad = 0.000
    │       ├── data = -0.111, grad = 0.000 ← weight[1]
    │       └── data = 1.000, grad = 0.000 ← x[1][0]
    └── data = 0.000, grad = 0.000 ← bias

Backward pass:
ReLU data = 1.800, grad = 1.000
└── + data = 1.800, grad = 1.000
    ├── + data = 1.800, grad = 1.000
    │   ├── * data = 1.911, grad = 1.000
    │   │   ├── data = 0.955, grad = 2.000 ← weight[0]
    │   │   └── data = 2.000, grad = 0.955 ← x[0][0]
    │   └── * data = -0.111, grad = 1.000
    │       ├── data = -0.111, grad = 1.000 ← weight[1]
    │       └── data = 1.000, grad = -0.111 ← x[1][0]
    └── data = 0.000, grad = 1.000 ← bias
```

---

##### MNIST

```rust
use ferrograd::{
    engine::{Activation, Value},
    nn::{softmax, MultiLayerPerceptron},
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

    let xtest: Vec<Vec<Value>> =
        images_to_features(&mnist.test_data[offset..offset + test_samples]);

    // Making predictions
    let mut correct = 0;
    xtest.iter().enumerate().for_each(|(i, x)| {
        let ypred = vec![model.forw(&x)];
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

// --- Helper functions ---
```

```console
cargo run --example test_mnist
```

```
# ...

□ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ 
□ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ 
□ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ 
□ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ 
□ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ 
□ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ ■ ■ ■ ■ ■ ■ ■ ■ ■ □ □ □ 
□ □ □ □ □ □ □ □ □ □ □ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ □ □ □ 
□ □ □ □ □ □ □ □ □ □ □ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ □ □ □ 
□ □ □ □ □ □ □ □ □ □ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ □ □ □ □ □ □ 
□ □ □ □ □ □ □ □ □ ■ ■ ■ ■ ■ □ ■ □ □ □ □ □ □ □ □ □ □ □ □ 
□ □ □ □ □ □ □ □ ■ ■ ■ ■ ■ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ 
□ □ □ □ □ □ □ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ □ □ □ □ □ □ □ □ □ □ □ 
□ □ □ □ □ □ □ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ □ □ □ □ □ □ □ □ □ 
□ □ □ □ □ □ □ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ □ □ □ □ □ □ □ □ 
□ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ ■ ■ ■ ■ ■ □ □ □ □ □ □ □ 
□ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ ■ ■ ■ ■ □ □ □ □ □ □ □ 
□ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ ■ ■ ■ ■ □ □ □ □ □ □ □ 
□ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ ■ ■ ■ ■ □ □ □ □ □ □ □ 
□ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ ■ ■ ■ ■ ■ □ □ □ □ □ □ □ 
□ □ □ □ □ ■ ■ ■ □ □ □ □ □ □ □ ■ ■ ■ ■ ■ ■ □ □ □ □ □ □ □ 
□ □ □ □ □ ■ ■ ■ □ □ □ □ □ ■ ■ ■ ■ ■ ■ ■ □ □ □ □ □ □ □ □ 
□ □ □ □ □ ■ ■ ■ ■ □ ■ ■ ■ ■ ■ ■ ■ ■ ■ □ □ □ □ □ □ □ □ □ 
□ □ □ □ □ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ □ □ □ □ □ □ □ □ □ □ 
□ □ □ □ □ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ □ □ □ □ □ □ □ □ □ □ □ 
□ □ □ □ □ □ ■ ■ ■ ■ ■ ■ ■ ■ ■ □ □ □ □ □ □ □ □ □ □ □ □ □ 
□ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ 
□ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ 
□ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ 
ytrue: 5
ypred: 5
prob: 0.822
pred: true
                                                        
# ...
```

---

##### scikit-learn's make_moons dataset classification

```rust
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

// fn print_grid(model: &MultiLayerPerceptron, bound: i32) { ... }
```

```console
cargo run --example moons
```

```
# ...
step 97 - loss 0.012, accuracy 100.00%
step 98 - loss 0.012, accuracy 100.00%
step 99 - loss 0.012, accuracy 100.00%

ASCII contour graph -
■ > 0.0
□ <= 0.0

□ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □
□ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □
□ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □
□ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □
□ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □
□ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □
□ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □
□ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □
□ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □
□ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ ■
□ □ □ □ □ □ □ □ □ □ □ □ □ □ □ ■ □ □ □ □ □ □ □ □ □ □ □ □ □ ■
□ □ □ □ □ □ □ □ □ □ □ □ □ ■ ■ ■ ■ ■ □ □ □ □ □ □ □ □ □ □ ■ ■
□ □ □ □ □ □ □ □ □ □ □ □ □ ■ ■ ■ ■ ■ ■ □ □ □ □ □ □ □ □ □ ■ ■
□ □ □ □ □ □ □ □ □ □ □ □ ■ ■ ■ ■ ■ ■ ■ ■ □ □ □ □ □ □ □ ■ ■ ■
□ □ □ □ □ □ □ □ □ □ □ □ ■ ■ ■ ■ ■ ■ ■ ■ ■ □ □ □ □ □ □ ■ ■ ■
□ □ □ □ □ □ □ □ □ □ □ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ □ □ □ □ ■ ■ ■ ■
□ □ □ □ □ □ □ □ □ □ □ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ □ ■ ■ ■ ■ ■
□ □ □ □ □ □ □ □ □ □ □ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■
□ □ □ □ □ □ □ □ □ □ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■
□ □ □ □ □ □ □ □ □ □ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■
□ □ □ □ □ □ □ □ □ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■
□ □ □ □ □ □ □ □ □ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■
□ □ □ □ □ □ □ □ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■
□ □ □ □ □ □ □ □ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■
□ □ □ □ □ □ □ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■
□ □ □ □ □ □ □ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■
□ □ □ □ □ □ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■
□ □ □ □ □ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■
□ □ □ □ □ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■
□ □ □ □ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■
```

> TODO:
> - AdamW, AdaGrad, RMSProp
> - Some performance optimisations
> - Documentation and notes

###### Credits

- [karpathy/micrograd](https://github.com/karpathy/micrograd)
- [Mathemmagician/rustygrad](https://github.com/Mathemmagician/rustygrad)
