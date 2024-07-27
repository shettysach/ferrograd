#### SGD with momentum 

$$
V_t = \beta V_{t-1} + \alpha \nabla_W L(W, X, y)
$$

$$
W = W - V_t
$$

```rust
pub fn step(&mut self) {
    self.params.iter().zip(self.velocities.iter_mut()).for_each(
        |(param, velocity)| {
            *velocity =
                self.momentum * *velocity + self.lr * param.borrow().grad;
            param.borrow_mut().data -= *velocity;
        },
    )
}
```
[Vitaly Bushaev's blog](https://towardsdatascience.com/stochastic-gradient-descent-with-momentum-a84097641a5d)

---

#### RMSprop

$$\begin{aligned}
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
\end{aligned}$$

$$
w = w_{t-1} - \frac{\eta}{\sqrt{\hat{v_t}} + \epsilon} g_t
$$

```rust
pub fn step(&mut self) {
    self.t += 1;

    self.params
        .iter()
        .zip(self.v.iter_mut())
        .for_each(|(param, v_t)| {
            let grad = param.borrow().grad;

            *v_t = self.beta * *v_t + (1.0 - self.beta) * grad.powi(2);

            param.borrow_mut().data -=
                self.lr / (v_t.sqrt() + self.epsilon) * grad;
        })
}
```

[Shuvam Das's blog](https://medium.com/deepkapha-notes/optimization-algorithms-and-interactive-visualization-part-2-4d6d9791e1d3)

---

#### Adam

$$\begin{aligned}
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t\\
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
\end{aligned}$$

$$\begin{aligned}
\hat{m_t} = \frac{m_t}{1 - \beta_1}\\
\hat{v_t} = \frac{v_t}{1 - \beta_2}
\end{aligned}$$

$$
w = w_{t-1} - \eta \frac{\hat{m_t}}{\sqrt{\hat{v_t}} + \epsilon}
$$

```rust
pub fn step(&mut self) {
    self.t += 1;

    self.params
        .iter()
        .zip(self.m.iter_mut().zip(self.v.iter_mut()))
        .for_each(|(param, (m_t, v_t))| {
            let grad = param.borrow().grad;

            *m_t = self.beta1 * *m_t + (1.0 - self.beta1) * grad;
            *v_t = self.beta2 * *v_t + (1.0 - self.beta2) * grad * grad;

            let mc_t = *m_t / (1.0 - self.beta1.powi(self.t as i32));
            let vc_t = *v_t / (1.0 - self.beta2.powi(self.t as i32));

            param.borrow_mut().data -=
                self.lr * mc_t / (vc_t.sqrt() + self.epsilon);
        })
}
```

[Vitaly Bushaev's blog](https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c)
