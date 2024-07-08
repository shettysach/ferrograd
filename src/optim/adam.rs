use crate::engine::Value;

pub struct Adam {
    params: Vec<Value>,
    lr: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    m: Vec<f64>,
    v: Vec<f64>,
    t: usize,
}

impl Adam {
    pub fn new(
        params: Vec<Value>,
        lr: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    ) -> Adam {
        let m = vec![0.0; params.len()];
        let v = vec![0.0; params.len()];

        Adam {
            params,
            lr,
            beta1,
            beta2,
            epsilon,
            m,
            v,
            t: 0,
        }
    }

    pub fn step(&mut self) {
        self.t += 1;
        let lr_t = self.lr * (1.0 - self.beta2.powi(self.t as i32)).sqrt()
            / (1.0 - self.beta1.powi(self.t as i32));

        for (param, (m, v)) in self
            .params
            .iter_mut()
            .zip(self.m.iter_mut().zip(self.v.iter_mut()))
        {
            let grad = param.borrow().grad;

            *m = self.beta1 * (*m + (1.0 - self.beta1) * grad);
            *v = self.beta2 * (*v + (1.0 - self.beta2) * grad * grad);

            let m_t = *m / (1.0 - self.beta1.powi(self.t as i32));
            let v_t = *v / (1.0 - self.beta2.powi(self.t as i32));

            param.borrow_mut().data -= lr_t * m_t / (v_t.sqrt() + self.epsilon);
        }
    }

    pub fn zero_grad(&self) {
        for p in &self.params {
            p.borrow_mut().grad = 0.0;
        }
    }
}
