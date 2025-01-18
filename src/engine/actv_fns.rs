use crate::engine::value::{ActvFn, Op, Prev, Value, V};

impl Value {
    pub fn relu(&self) -> Value {
        Value::init(
            self.borrow().data.max(0.0),
            Some(|value: &V| {
                if let Prev::Unary(a) = &value.prev {
                    a.borrow_mut().grad += if value.data > 0.0 { value.grad } else { 0.0 };
                }
            }),
            Prev::Unary(self.clone()),
            Op::ActvFn(ActvFn::ReLU),
            None,
        )
    }

    pub fn leaky_relu(&self) -> Value {
        let x = self.borrow().data;
        Value::init(
            x.max(0.01 * x),
            Some(|value: &V| {
                if let Prev::Unary(a) = &value.prev {
                    a.borrow_mut().grad += if value.data > 0.0 {
                        value.grad
                    } else {
                        0.01 * value.grad
                    };
                }
            }),
            Prev::Unary(self.clone()),
            Op::ActvFn(ActvFn::LeakyReLU),
            None,
        )
    }

    pub fn tanh(&self) -> Value {
        let e2x = (2.0 * self.borrow().data).exp();
        Value::init(
            (e2x - 1.0) / (e2x + 1.0),
            Some(|value: &V| {
                if let Prev::Unary(a) = &value.prev {
                    a.borrow_mut().grad += (1.0 - value.data.powi(2)) * value.grad;
                }
            }),
            Prev::Unary(self.clone()),
            Op::ActvFn(ActvFn::Tanh),
            None,
        )
    }

    pub fn sigmoid(&self) -> Value {
        let em1x = (-1.0 * self.borrow().data).exp();
        Value::init(
            1.0 / (1.0 + em1x),
            Some(|value: &V| {
                if let Prev::Unary(a) = &value.prev {
                    a.borrow_mut().grad += value.data * (1.0 - value.data) * value.grad;
                }
            }),
            Prev::Unary(self.clone()),
            Op::ActvFn(ActvFn::Sigmoid),
            None,
        )
    }
}
