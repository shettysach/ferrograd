### micrograd

- A small autograd engine, inspired from [karpathy/micrograd](https://github.com/karpathy/micrograd), with a few more features, such as additional activation functions, loss criterions, optimizers and accuracy metrics.
- See `/notes/Gradients.md` for explanation of gradients and backward functions, and `/notes/Optimizers.md` for the equations and step functions of optimizers.
- The library lets you create neurons, dense layers and multilayer perceptrons, for non-linear classification tasks.
- Currently has criterions and metrics for binary classification.

#### Examples

##### Readme example from karpathy/micrograd

```rust
use micrograd::engine::Value;

fn main() {
    let a = Value::new(-4.).with_name("a");
    let b = Value::new(2.).with_name("b");

    let mut c = (&a + &b).with_name("c");

    let mut d = (&a * &b + &b.pow(3.)).with_name("c");

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
g.data = 24.7041
a.grad = 138.8338
b.grad = 645.5773
```

---

##### Neuron

```rust
use micrograd::{
    engine::{Activation, Value},
    nn::Neuron,
};

fn main() {
    let n = Neuron::new(2, Some(Activation::Sigmoid)).name_params();

    let x = vec![Value::new(-2.0), Value::new(1.0)];
    let x = n.name_inputs(x);

    println!("{}\n", n);

    let y = n.forward(&x);
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
ReLU data = 0.000, grad = 0.000 
└── + data = -1.159, grad = 0.000 
    ├── + data = -1.159, grad = 0.000 
    │   ├── * data = -0.543, grad = 0.000 
    │   │   ├── data = 0.272, grad = 0.000 ← weight[0]
    │   │   └── data = -2.000, grad = 0.000 ← x[0][0]
    │   └── * data = -0.615, grad = 0.000 
    │       ├── data = -0.615, grad = 0.000 ← weight[1]
    │       └── data = 1.000, grad = 0.000 ← x[1][0]
    └── data = 0.000, grad = 0.000 ← bias

Backward pass:
ReLU data = 0.000, grad = 1.000 
└── + data = -1.159, grad = 0.000 
    ├── + data = -1.159, grad = 0.000 
    │   ├── * data = -0.543, grad = 0.000 
    │   │   ├── data = 0.272, grad = 0.000 ← weight[0]
    │   │   └── data = -2.000, grad = 0.000 ← x[0][0]
    │   └── * data = -0.615, grad = 0.000 
    │       ├── data = -0.615, grad = 0.000 ← weight[1]
    │       └── data = 1.000, grad = 0.000 ← x[1][0]
    └── data = 0.000, grad = 0.000 ← bias
```

---

##### scikit-learn's make_moons dataset classification

```rust
use micrograd::{
    engine::{Activation, Value},
    loss::HingeLoss,
    metrics::BinaryAccuracy,
    nn::{
        optim::{l2_regularization, Adam},
        MultiLayerPerceptron,
    },
};

fn main() {
    let (xs, ys) = load_data();
    let model = MultiLayerPerceptron::new(2, vec![16, 16, 1], Activation::ReLU);

    println!("Model - \n{}", model);
    println!("Number of parameters = {}\n", model.parameters().len());

    let mut optim = Adam::new(model.parameters(), 0.1, 0.9, 0.999, 0.00000001);
    let loss = HingeLoss::new(1.0);
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
}
```

```console
cargo run --example moons
```

```
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

###### Credits

- [karpathy/micrograd](https://github.com/karpathy/micrograd)
- [Mathemmagician/rustygrad](https://github.com/Mathemmagician/rustygrad)
