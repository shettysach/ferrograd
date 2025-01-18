use crate::engine::Value;
use std::fmt;

pub struct SGD {
    params: Vec<Value>,
    lr: f64,
    momentum: f64,
    velocities: Vec<f64>,
}

impl SGD {
    pub fn new(params: Vec<Value>, lr: f64, momentum: f64) -> SGD {
        let velocities = vec![0.0; params.len()];

        SGD {
            params,
            lr,
            momentum,
            velocities,
        }
    }

    pub fn step(&mut self) {
        for (param, velocity) in self.params.iter().zip(self.velocities.iter_mut()) {
            *velocity = self.momentum * (*velocity) + self.lr * param.borrow().grad;
            param.borrow_mut().data -= *velocity;
        }
    }

    pub fn zero_grad(&self) {
        for p in &self.params {
            p.borrow_mut().grad = 0.0;
        }
    }
}

impl fmt::Debug for SGD {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SGD")
            .field("lr", &self.lr)
            .field("momentum", &self.momentum)
            .finish()
    }
}
