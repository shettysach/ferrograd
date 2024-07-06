use crate::engine::Value;

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
        for (param, velocity) in
            self.params.iter_mut().zip(self.velocities.iter_mut())
        {
            *velocity =
                self.momentum * *velocity - self.lr * param.borrow().grad;
            param.borrow_mut().data += *velocity;
        }
    }

    pub fn zero_grad(&self) {
        for p in &self.params {
            p.borrow_mut().grad = 0.0;
        }
    }
}
