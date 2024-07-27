#### Mean Square Error Loss

$$
MSE = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y_i})^2 
$$

```rust
pub fn loss(
    &self,
    ypred: &Vec<Vec<Value>>,
    ytrue: &Vec<Vec<Value>>,
) -> Value {
    ypred
        .iter()
        .zip(ytrue)
        .map(|(ypred_i, ytrue_i)| {
            ypred_i
                .iter()
                .zip(ytrue_i)
                .map(|(ypred_j, ytrue_j)| (ytrue_j - ypred_j).pow(2.0))
                .sum::<Value>()
                / ypred_i.len() as f64
        })
        .sum::<Value>()
        / ypred.len() as f64
}
```

---

#### Hinge loss

$$
L = \frac{1}{N} \sum_{i=1}^N \max(0, 1 - y_i\cdot\hat{y_i}) 
$$

```rust
pub fn loss(
    &self,
    ypred: &Vec<Vec<Value>>,
    ytrue: &Vec<Vec<Value>>,
) -> Value {
    ypred
        .iter()
        .zip(ytrue.iter())
        .map(|(ypred_i, ytrue_i)| {
            ypred_i
                .iter()
                .zip(ytrue_i)
                .map(|(ypred_j, ytrue)| (1.0 - ytrue * ypred_j).relu())
                .sum::<Value>()
        })
        .sum::<Value>()
        / (ypred.len() * ypred[0].len()) as f64
}
```

---

#### Cross-Entropy loss

$$
L = - \frac{1}{N} \sum_{i=1}^N y_i \cdot \log{\hat{y_i}}
$$

```rust
pub fn loss(
    &self,
    ypred: &Vec<Vec<Value>>,
    ytrue: &Vec<Vec<Value>>,
) -> Value {
    -ypred
        .iter()
        .zip(ytrue)
        .map(|(ypred_i, ytrue_i)| {
            ypred_i
                .iter()
                .zip(ytrue_i)
                .map(|(ypred_j, ytrue_j)| ytrue_j * ypred_j.ln())
                .sum::<Value>()
        })
        .sum::<Value>()
        / (ypred.len() * ypred[0].len()) as f64
}
```

---

#### Binary Cross-Entropy loss

$$
L = - \frac{1}{N} \sum_{i=1}^N y_i \cdot \log{\hat{y_i}} +  (1 - y_i) \cdot \log{(1 - \hat{y_i})}
$$

```rust
pub fn loss(
    &self,
    ypred: &Vec<Vec<Value>>,
    ytrue: &Vec<Vec<Value>>,
) -> Value {
    -ypred
        .iter()
        .zip(ytrue)
        .map(|(ypred_i, ytrue_i)| {
            ypred_i
                .iter()
                .zip(ytrue_i)
                .map(|(ypred_j, ytrue_j)| {
                    ytrue_j * ypred_j.ln()
                        + (1.0 - ytrue_j) * (1.0 - ypred_j).ln()
                })
                .sum::<Value>()
        })
        .sum::<Value>()
        / (ypred.len() * ypred[0].len()) as f64
}
```
