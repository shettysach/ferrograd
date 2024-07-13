use crate::engine::Value;

// See /notes/Optimizers.md
// https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c

/// Adam optimizer.
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
    /// Initialise new optimizer.
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

    /// Perform a single optimizer step.
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

    /// Set the gradients of all the parameters to zero.
    pub fn zero_grad(&self) {
        for p in &self.params {
            p.borrow_mut().grad = 0.0;
        }
    }
}
