### micrograd

- A small autograd engine, inspired from [karpathy/micrograd](https://github.com/karpathy/micrograd), with a few more features, such as additional activation functions, loss criterions, optimizers and accuracy metrics.
- See `/notes/Gradients.md` for explanation of gradients and backward functions, and `/notes/Optimizers.md` for the equations and step functions of optimizers.
- The library lets you create neurons, dense layers and multilayer perceptrons, for non-linear classification tasks.
- Currently only has criterions and metrics for binary classification. 
> TODO: Multiclass classification support.

- Readme example from karpathy/micrograd

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
- scikit-learn's make_moons dataset classification

```rust
use micrograd::{
    engine::{Activation, Value},
    metrics::BinaryAccuracy,
    nn::{
        loss::HingeEmbeddingLoss,
        optim::{adam::Adam, l2_regularization},
        MultiLayerPerceptron,
    },
};

fn main() {
    let model = MultiLayerPerceptron::new(2, vec![16, 16, 1], Activation::ReLU);

    println!("Model - \n{}", model);
    println!("Number of parameters = {}\n", model.parameters().len());

    let (xs, ys) = load_moons_data();
    let mut optim = Adam::new(model.parameters(), 0.1, 0.9, 0.999, 0.00000001);
    let loss = HingeEmbeddingLoss::new(1.0);
    let accuracy = BinaryAccuracy::new(0.0);

    (0..100).for_each(|k| {
        let ypred: Vec<Vec<Value>> =
            xs.iter().map(|xrow| model.forward(&xrow)).collect();

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

##### Examples

```console
cargo run --example neuron
```

```
Forward pass:
R data = 0.326, grad = 0.000
└── + data = 0.326, grad = 0.000
    ├── + data = 0.326, grad = 0.000
    │   ├── * data = 0.422, grad = 0.000
    │   │   ├── data = -0.211, grad = 0.000 ← weight 0
    │   │   └── data = -2.000, grad = 0.000 ← input 0
    │   └── * data = -0.096, grad = 0.000
    │       ├── data = -0.096, grad = 0.000 ← weight 1
    │       └── data = 1.000, grad = 0.000 ← input 1
    └── data = 0.000, grad = 0.000 ← bias

Backward pass:
R data = 0.326, grad = 1.000
└── + data = 0.326, grad = 1.000
    ├── + data = 0.326, grad = 1.000
    │   ├── * data = 0.422, grad = 1.000
    │   │   ├── data = -0.211, grad = -2.000 ← weight 0
    │   │   └── data = -2.000, grad = -0.211 ← input 0
    │   └── * data = -0.096, grad = 1.000
    │       ├── data = -0.096, grad = 1.000 ← weight 1
    │       └── data = 1.000, grad = -0.096 ← input 1
    └── data = 0.000, grad = 1.000 ← bias
```

---

```console
cargo run --example moons_adam
```

```
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

###### Credits

- [karpathy/micrograd](https://github.com/karpathy/micrograd)
- [Mathemmagician/rustygrad](https://github.com/Mathemmagician/rustygrad)
