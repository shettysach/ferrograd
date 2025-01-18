### ferrograd

- A small autograd engine, inspired from [karpathy/micrograd](https://github.com/karpathy/micrograd), with more features, such as additional activation functions, optimizers, loss criterions, accuracy metrics, and the ability to save and load model weights after training.
- Capable of creating neurons, dense layers and multilayer perceptrons, for non-linear classification tasks.

```console
cargo add --git https://github.com/shettysach/ferrograd.git ferrograd
```

#### Examples

##### Readme example from karpathy/micrograd

```rust
use ferrograd::engine::Value;

fn main() {
    let a = Value::new(-4.);
    let b = Value::new(2.);

    let mut c = &a + &b;
    let mut d = &a * &b + &b.pow(3.);

    c += &c + 1.;
    c += 1. + &c + (-&a);
    d += &d * 2. + (&b + &a).relu();
    d += 3. * &d + (&b - &a).relu();

    let e = &c - &d;
    let f = e.pow(2.);

    let mut g = &f / 2.;
    g += 10. / &f;

    println!("g.data = {:.4}", g.borrow().data);
    g.backward();

    println!("a.grad = {:.4}", a.borrow().grad);
    println!("b.grad = {:.4}", b.borrow().grad);
}
```

```console
cargo run --example readme
```

```
g.data = 24.7041
a.grad = 138.8338
b.grad = 645.5773
```

---

##### Expression tree

```rust
use ferrograd::engine::Value;

fn main() {
    let a = Value::new_with_name(5.6, 'a');
    let b = Value::new_with_name(10.8, 'b');

    let c = Value::new_with_name(-15.12, 'c');
    let d = Value::new_with_name(2.5, 'd');

    let e = (&a + &b) / 50.0;
    let e = e.with_name('e');

    let f = (&d - &c) * 5.5625;
    let f = f.with_name('f');

    let g = (e * f).leaky_relu().with_name('g');
    g.backward();

    println!("{}", g.tree());
}
```

```console
cargo run --example tree
```

```
LeakyReLU data = 32.148, grad = 1.000 ← g
└── * data = 32.148, grad = 1.000
    ├── * data = 0.328, grad = 98.011 ← e
    │   ├── + data = 16.400, grad = 1.960
    │   │   ├── data = 5.600, grad = 1.960 ← a
    │   │   └── data = 10.800, grad = 1.960 ← b
    │   └── 0.020
    └── * data = 98.011, grad = 0.328 ← f
        ├── + data = 17.620, grad = 1.824
        │   ├── data = 2.500, grad = 1.824 ← d
        │   └── * data = 15.120, grad = 1.824
        │       ├── data = -15.120, grad = -1.824 ← c
        │       └── -1.000
        └── 5.562
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
ReLU data = 0.562, grad = 0.000
└── + data = 0.562, grad = 0.000
    ├── + data = 0.562, grad = 0.000
    │   ├── * data = 1.360, grad = 0.000
    │   │   ├── data = 0.680, grad = 0.000 ← w
    │   │   └── data = 2.000, grad = 0.000 ← X
    │   └── * data = -0.798, grad = 0.000
    │       ├── data = -0.798, grad = 0.000 ← w
    │       └── data = 1.000, grad = 0.000 ← X
    └── data = 0.000, grad = 0.000 ← b

Backward pass: 
ReLU data = 0.562, grad = 1.000
└── + data = 0.562, grad = 1.000
    ├── + data = 0.562, grad = 1.000
    │   ├── * data = 1.360, grad = 1.000
    │   │   ├── data = 0.680, grad = 2.000 ← w
    │   │   └── data = 2.000, grad = 0.680 ← X
    │   └── * data = -0.798, grad = 1.000
    │       ├── data = -0.798, grad = 1.000 ← w
    │       └── data = 1.000, grad = -0.798 ← X
    └── data = 0.000, grad = 1.000 ← b
```

---

##### MNIST

> **NOTE:** To run the MNIST examples, download the [data](https://yann.lecun.com/exdb/mnist/), gzip extract all 4 files, and move them to the directory `/data/mnist/`.

```rust
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
```

```console
cargo run --example test_mnist
```

<div "align: center;">
    <img src="mnist.gif" width="75%">
</div>

---

##### scikit-learn's make_moons dataset classification

```rust
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

---

> NOTE:
>
> - Created for learning and not optimized for performance.
>   - Uses scalar values (`Value`) and operations. `Vec<Value>` and `Vec<Vec<Value>>` are used in place of 1d and 2d tensors.
>   - Negation and subtraction involves multiplication with -1 and division involves raising to the power -1, instead of direct implementations, similar to how it is implemented in micrograd.
> - Run examples with the `release` flag (`cargo run --release --example <example>`) for more better performance.

```
Optimizers
├ Adam
├ RMSprop
└ SGD with momentum

Loss criterions
├ Binary Cross-Entropy
├ Cross-Entropy
└ Hinge

Activation functions
├ Leaky ReLU
├ ReLU
├ Sigmoid
├ Softmax
└ Tanh

Accuracy metrics
└ Binary accuracy
```
###### Credits

- [karpathy/micrograd](https://github.com/karpathy/micrograd)
- [Mathemmagician/rustygrad](https://github.com/Mathemmagician/rustygrad)
