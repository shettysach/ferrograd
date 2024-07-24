use crate::engine::Value;
use std::fmt;

/// Stochastic Gradient Descent optimizer with momentum.
pub struct SGD {
    params: Vec<Value>,
    lr: f64,
    momentum: f64,
    velocities: Vec<f64>,
}

impl SGD {
    /// Initialise new optimizer.
    pub fn new(params: Vec<Value>, lr: f64, momentum: f64) -> SGD {
        let velocities = vec![0.0; params.len()];

        SGD {
            params,
            lr,
            momentum,
            velocities,
        }
    }

    /// Perform a single optimizer step.
    pub fn step(&mut self) {
        self.params.iter().zip(self.velocities.iter_mut()).for_each(
            |(param, velocity)| {
                *velocity =
                    self.momentum * *velocity + self.lr * param.borrow().grad;
                param.borrow_mut().data -= *velocity;
            },
        )
    }

    /// Set the gradients of all the parameters to zero.
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
