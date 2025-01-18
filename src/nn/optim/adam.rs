use crate::engine::Value;
use std::fmt;

/// Adaptive Moment Estimation.
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
    pub fn new(params: Vec<Value>, lr: f64, beta1: f64, beta2: f64, epsilon: f64) -> Adam {
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

        for (param, (m_t, v_t)) in self
            .params
            .iter()
            .zip(self.m.iter_mut().zip(self.v.iter_mut()))
        {
            let grad = param.borrow().grad;

            *m_t = self.beta1 * *m_t + (1.0 - self.beta1) * grad;
            *v_t = self.beta2 * *v_t + (1.0 - self.beta2) * grad.powi(2);

            let mc_t = *m_t / (1.0 - self.beta1.powi(self.t as i32));
            let vc_t = *v_t / (1.0 - self.beta2.powi(self.t as i32));

            param.borrow_mut().data -= self.lr * mc_t / (vc_t.sqrt() + self.epsilon);
        }
    }

    pub fn zero_grad(&self) {
        for p in &self.params {
            p.borrow_mut().grad = 0.0;
        }
    }
}

impl fmt::Debug for Adam {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Adam")
            .field("lr", &self.lr)
            .field("beta1", &self.beta1)
            .field("beta2", &self.beta2)
            .field("epsilon", &self.epsilon)
            .finish()
    }
}
