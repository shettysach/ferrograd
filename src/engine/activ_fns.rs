use super::{Activation, Operation, Value, V};

impl Value {
    pub fn relu(&self) -> Value {
        Value::init(
            self.borrow().data.max(0.0),
            Some(|value: &V| {
                value._prev[0].borrow_mut().grad +=
                    if value.data > 0.0 { value.grad } else { 0.0 };
            }),
            vec![self.clone()],
            Some(Operation::AF(Activation::ReLU)),
            Some(String::new()),
        )
    }

    pub fn leaky_relu(&self) -> Value {
        let x = self.borrow().data;
        Value::init(
            x.max(0.01 * x),
            Some(|value: &V| {
                value._prev[0].borrow_mut().grad += if value.data > 0.0 {
                    value.grad
                } else {
                    0.01 * value.grad
                };
            }),
            vec![self.clone()],
            Some(Operation::AF(Activation::LeakyReLU)),
            Some(String::new()),
        )
    }

    pub fn tanh(&self) -> Value {
        let e2x = (2.0 * self.borrow().data).exp();
        Value::init(
            (e2x - 1.0) / (e2x + 1.0),
            Some(|value: &V| {
                value._prev[0].borrow_mut().grad +=
                    (1.0 - (value.data.powi(2))) * value.grad;
            }),
            vec![self.clone()],
            Some(Operation::AF(Activation::Tanh)),
            Some(String::new()),
        )
    }

    pub fn sigmoid(&self) -> Value {
        let em1x = (-1.0 * self.borrow().data).exp();
        Value::init(
            1.0 / (1.0 + em1x),
            Some(|value: &V| {
                value._prev[0].borrow_mut().grad +=
                    value.data * (1.0 - value.data) * value.grad;
            }),
            vec![self.clone()],
            Some(Operation::AF(Activation::Sigmoid)),
            Some(String::new()),
        )
    }
}
