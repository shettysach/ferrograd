##### Chain rule

$$\begin{aligned}
s = f(z)\\
z = g(x)
\\\\
\frac{\partial{s}}{\partial{x}}=
\frac{\partial{z}}{\partial{x}}\cdot
\frac{\partial{s}}{\partial{z}}
\end{aligned}$$

x.grad += ∂z/∂x * z.grad

----

##### Addition

$$\begin{aligned}
z = x+y
\\\\
\frac{\partial{z}}{\partial{x}} = 1
\end{aligned}$$

x.grad += z.grad

```python
def _backward():
    self.grad += out.grad
    other.grad += out.grad
```

```rust
fn add_backward(value: &V) {
    value._prev[0].borrow_mut().grad += value.grad;
    value._prev[1].borrow_mut().grad += value.grad;
}
```
----

##### Multiplication

$$\begin{aligned}
z = xy
\\\\
\frac{\partial{z}}{\partial{x}} = y
\end{aligned}$$

x.grad += y.data * z.grad

```python
def _backward():
    self.grad += other.data * out.grad
    other.grad += self.data * out.grad
```

```rust
fn mul_backward(value: &V) {
    let data0 = value._prev[0].borrow().data;
    let data1 = value._prev[1].borrow().data;
    value._prev[0].borrow_mut().grad += data1 * value.grad;
    value._prev[1].borrow_mut().grad += data0 * value.grad;
}
```
----

##### Power

$$\begin{aligned}
z = x^y
\\\\
\frac{\partial{z}}{\partial{x}} = yx^{y-1}
\end{aligned}$$

x.grad += (y.data \* x.data ** (y.data - 1)) \* z.grad

```python        
def _backward():
    self.grad += (other * self.data**(other-1)) * out.grad
```

```rust
Some(|value: &V| {
    let base = value._prev[0].borrow().data;
    let power = value._prev[1].borrow().data;
    value._prev[0].borrow_mut().grad += 
        power * base.powf(power - 1.0) * value.grad;
}),
```
----

##### ReLU - Rectified Linear Unit

$$\begin{aligned}
z = \max(0,x)
\\\\
\frac{\partial{z}}{\partial{x}} = 
    \begin{array}{ll}
    1, & \text{if } x > 0,\\
    0, & \text{otherwise.}
    \end{array}
\end{aligned}$$

x.grad += z.grad if (x.data > 0) else 0

```python        
def _backward():
    self.grad += (out.data > 0) * out.grad
```

```rust
Some(|value: &V| {
    value._prev[0].borrow_mut().grad += 
        if value.data > 0.0 { value.grad } else { 0.0 };
}),
```
----

##### Leaky ReLU

$$\begin{aligned}
z = \max(0.01x,x)
\\\\
\frac{\partial{z}}{\partial{x}} = 
    \begin{array}{ll}
    1, & \text{if } x > 0,\\
    0.01, & \text{otherwise.}
    \end{array}
\end{aligned}$$

```rust
Some(|value: &V| {
    value._prev[0].borrow_mut().grad += if value.data > 0.0 {
        value.grad
    } else {
        0.01 * value.grad
    };
}),
```
----

##### Hyperbolic tangent

$$\begin{aligned}
z = \tanh x = \frac{e^{2x}-1}{e^{2x}+1} 
\\\\
\frac{\partial{z}}{\partial{x}} = 1 - \tanh^2 x 
\end{aligned}$$

x.grad += (1 - z.data ^ 2) * z.grad

```rust
Some(|value: &V| {
    value._prev[0].borrow_mut().grad +=
        (1. - (value.data.powi(2))) * value.grad;
}),
```
----

##### Sigmoid

$$\begin{aligned}
z = \sigma(x) = \frac{1}{1 + e^{-x}} 
\\\\
\frac{\partial{z}}{\partial{x}} = \sigma(x) \cdot (1 - \sigma(x)) 
\end{aligned}$$

x.grad += z.data \* (1 - z.data) \* z.grad

```rust
Some(|value: &V| {
    value._prev[0].borrow_mut().grad += 
        value.data * (1. - value.data) * value.grad;
}),
```
----

##### Negation

$$\begin{aligned}
z = -1\cdot y
\\\\
\frac{\partial{z}}{\partial{y}} = -1
\end{aligned}$$

y.grad += -1 \* z.grad

----

##### Subtraction

$$\begin{aligned}
z = x-y
\\\\
\frac{\partial{z}}{\partial{y}} = -1
\end{aligned}$$

y.grad += -1 \* z.grad

----

##### Division

$$\begin{aligned}
z = x/y
\\\\
\frac{\partial{z}}{\partial{y}} = -xy^{-2}
\end{aligned}$$

y,grad += -1 \* x.data \* y.data ** (-2) \* z.grad

----
