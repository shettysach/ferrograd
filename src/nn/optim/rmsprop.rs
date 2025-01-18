use crate::engine::Value;
use std::fmt;

pub struct RMSprop {
    params: Vec<Value>,
    lr: f64,
    beta: f64,
    epsilon: f64,
    v: Vec<f64>,
    t: usize,
}

impl RMSprop {
    pub fn new(params: Vec<Value>, lr: f64, beta: f64, epsilon: f64) -> RMSprop {
        let v = vec![0.0; params.len()];

        RMSprop {
            params,
            lr,
            beta,
            epsilon,
            v,
            t: 0,
        }
    }

    pub fn step(&mut self) {
        self.t += 1;

        for (param, v_t) in self.params.iter().zip(self.v.iter_mut()) {
            let grad = param.borrow().grad;

            *v_t = self.beta * *v_t + (1.0 - self.beta) * grad.powi(2);

            param.borrow_mut().data -= self.lr / (v_t.sqrt() + self.epsilon) * grad;
        }
    }

    pub fn zero_grad(&self) {
        for p in &self.params {
            p.borrow_mut().grad = 0.0;
        }
    }
}

impl fmt::Debug for RMSprop {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RMSprop")
            .field("lr", &self.lr)
            .field("beta", &self.beta)
            .field("epsilon", &self.epsilon)
            .finish()
    }
}
