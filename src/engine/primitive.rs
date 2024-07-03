use super::{Activation, Operation, Value, V};
use std::{f64::consts::E, ops};

// ADDITION

#[opimps::impl_ops(ops::Add)]
fn add(self: Value, rhs: Value) -> Value {
    Value::init(
        self.borrow().data + rhs.borrow().data,
        Some(add_backward),
        vec![self.clone(), rhs.clone()],
        Some(Operation::Add),
        Some(String::new()),
    )
}

#[opimps::impl_ops_rprim(ops::Add)]
fn add(self: Value, rhs: f64) -> Value {
    Value::init(
        self.borrow().data + rhs,
        Some(add_backward),
        vec![self.clone(), Value::new_const(rhs)],
        Some(Operation::Add),
        Some(String::new()),
    )
}

#[opimps::impl_ops_lprim(ops::Add)]
fn add(self: f64, rhs: Value) -> Value {
    Value::init(
        self + rhs.borrow().data,
        Some(add_backward),
        vec![Value::new_const(self), rhs.clone()],
        Some(Operation::Add),
        Some(String::new()),
    )
}

fn add_backward(value: &V) {
    value.prev[0].borrow_mut().grad += value.grad;
    value.prev[1].borrow_mut().grad += value.grad;
}

// MULTIPLICATION

#[opimps::impl_ops(ops::Mul)]
fn mul(self: Value, rhs: Value) -> Value {
    Value::init(
        self.borrow().data * rhs.borrow().data,
        Some(mul_backward),
        vec![self.clone(), rhs.clone()],
        Some(Operation::Mul),
        Some(String::new()),
    )
}

#[opimps::impl_ops_rprim(ops::Mul)]
fn mul(self: Value, rhs: f64) -> Value {
    Value::init(
        self.borrow().data * rhs,
        Some(mul_backward),
        vec![self.clone(), Value::new_const(rhs)],
        Some(Operation::Mul),
        Some(String::new()),
    )
}

#[opimps::impl_ops_lprim(ops::Mul)]
fn mul(self: f64, rhs: Value) -> Value {
    Value::init(
        self * rhs.borrow().data,
        Some(mul_backward),
        vec![Value::new_const(self), rhs.clone()],
        Some(Operation::Mul),
        Some(String::new()),
    )
}

fn mul_backward(value: &V) {
    let data0 = value.prev[0].borrow().data;
    let data1 = value.prev[1].borrow().data;
    value.prev[0].borrow_mut().grad += data1 * value.grad;
    value.prev[1].borrow_mut().grad += data0 * value.grad;
}

// Power and Activation functions

impl Value {
    pub fn pow(&self, power: f64) -> Value {
        Value::init(
            self.borrow().data.powf(power),
            Some(|value: &V| {
                let base = value.prev[0].borrow().data;
                let power = value.prev[1].borrow().data;
                value.prev[0].borrow_mut().grad +=
                    power * base.powf(power - 1.0) * value.grad;
            }),
            vec![self.clone(), Value::new_const(power)],
            Some(Operation::Pow),
            Some(String::new()),
        )
    }

    pub fn relu(&self) -> Value {
        Value::init(
            self.borrow().data.max(0.0),
            Some(|value: &V| {
                value.prev[0].borrow_mut().grad +=
                    if value.data > 0.0 { value.grad } else { 0.0 };
            }),
            vec![self.clone()],
            Some(Operation::AF(Activation::ReLU)),
            Some(String::new()),
        )
    }

    pub fn tanh(&self) -> Value {
        let e2x = E.powf(2. * self.borrow().data);
        Value::init(
            (e2x - 1.) / (e2x + 1.),
            Some(|value: &V| {
                value.prev[0].borrow_mut().grad +=
                    (1. - (value.data.powi(2))) * value.grad;
            }),
            vec![self.clone()],
            Some(Operation::AF(Activation::Tanh)),
            Some(String::new()),
        )
    }

    pub fn sigmoid(&self) -> Value {
        let enx = E.powf(-1. * self.borrow().data);
        Value::init(
            1. / (1. + enx),
            Some(|value: &V| {
                value.prev[0].borrow_mut().grad +=
                    value.data * (1. - value.data) * value.grad;
            }),
            vec![self.clone()],
            Some(Operation::AF(Activation::Sigmoid)),
            Some(String::new()),
        )
    }

    fn new_const(data: f64) -> Value {
        Value::init(data, None, Vec::new(), None, None)
    }
}
