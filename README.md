### ferrograd

- A small autograd engine, inspired from [karpathy/micrograd](https://github.com/karpathy/micrograd), with a few more features, such as additional activation functions, loss criterions, optimizers and accuracy metrics.
- See `/notes/Gradients.md` for explanation of gradients and backward functions, and `/notes/Optimizers.md` for the equations and step functions of optimizers.
- Capable of creating neurons, dense layers and multilayer perceptrons, for non-linear classification tasks.

> NOTE: WIP

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
    let n = Neuron::new(2, Some(Activation::ReLU)).name_params();

    let x = vec![
        vec![Value::new(-2.0)], // x0
        vec![Value::new(1.0)],  // x1
    ];
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

---

##### scikit-learn's make_circles dataset classification

```rust
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
    let (xs, ys) = read_csv("data/circles_data.csv", 2, 1, 1);

    let model = MultiLayerPerceptron::new(2, vec![16, 16, 1], Activation::ReLU);
    println!("Model - \n{}", model);
    println!("Number of parameters = {}\n", model.parameters().len());

    let mut optim = Adam::new(model.parameters(), 0.1, 0.9, 0.999, 1e-7);
    let loss = BinaryCrossEntropyLoss::new();
    let accuracy = BinaryAccuracy::new(0.5);

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
cargo run --example circles
```

```
# ...
step 97 - loss 0.022, accuracy 100.00%
step 98 - loss 0.022, accuracy 100.00%
step 99 - loss 0.021, accuracy 100.00%

ASCII contour graph - 
■ > 0.5  
□ <= 0.5 

□ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □
□ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □
□ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □
□ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □
□ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □
□ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □
□ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □
□ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □
□ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □
□ □ □ □ □ □ □ □ □ □ □ □ ■ ■ ■ ■ ■ ■ ■ □ □ □ □ □ □ □ □ □ □ □
□ □ □ □ □ □ □ □ □ □ □ ■ ■ ■ ■ ■ ■ ■ ■ ■ □ □ □ □ □ □ □ □ □ □
□ □ □ □ □ □ □ □ □ □ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ □ □ □ □ □ □ □ □ □
□ □ □ □ □ □ □ □ □ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ □ □ □ □ □ □ □ □
□ □ □ □ □ □ □ □ □ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ □ □ □ □ □ □ □ □
□ □ □ □ □ □ □ □ □ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ □ □ □ □ □ □ □ □
□ □ □ □ □ □ □ □ □ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ □ □ □ □ □ □ □ □
□ □ □ □ □ □ □ □ □ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ □ □ □ □ □ □ □ □
□ □ □ □ □ □ □ □ □ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ □ □ □ □ □ □ □ □
□ □ □ □ □ □ □ □ □ □ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ □ □ □ □ □ □ □ □ □
□ □ □ □ □ □ □ □ □ □ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ □ □ □ □ □ □ □ □ □
□ □ □ □ □ □ □ □ □ □ □ ■ ■ ■ ■ ■ ■ ■ ■ ■ □ □ □ □ □ □ □ □ □ □
□ □ □ □ □ □ □ □ □ □ □ □ ■ ■ ■ ■ ■ ■ ■ □ □ □ □ □ □ □ □ □ □ □
□ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □
□ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □
□ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □
□ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □
□ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □
□ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □
□ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □
□ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □
```

---

> TODO: 
> - Multiclass classification support
> - Multiclass and Multilabel cross-entropy loss
> - AdaGrad, RMSProp and AdamW
> - Documentation and notes


###### Credits

- [karpathy/micrograd](https://github.com/karpathy/micrograd)
- [Mathemmagician/rustygrad](https://github.com/Mathemmagician/rustygrad)
