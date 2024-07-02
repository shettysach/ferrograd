use super::{Operation, Value, V};
use std::ops;

// ADDITION

#[opimps::impl_ops(ops::Add)]
fn add(self: Value, rhs: Value) -> Value {
    Value::init(
        self.borrow().data + rhs.borrow().data,
        0.0,
        Some(add_backward),
        vec![self.clone(), rhs.clone()],
        Some(Operation::Add),
    )
}

#[opimps::impl_ops_rprim(ops::Add)]
fn add(self: Value, rhs: f64) -> Value {
    Value::init(
        self.borrow().data + rhs,
        0.0,
        Some(add_backward),
        vec![self.clone(), Value::new(rhs)],
        Some(Operation::Add),
    )
}

#[opimps::impl_ops_lprim(ops::Add)]
fn add(self: f64, rhs: Value) -> Value {
    Value::init(
        self + rhs.borrow().data,
        0.0,
        Some(add_backward),
        vec![Value::new(self), rhs.clone()],
        Some(Operation::Add),
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
        0.0,
        Some(mul_backward),
        vec![self.clone(), rhs.clone()],
        Some(Operation::Mul),
    )
}

#[opimps::impl_ops_rprim(ops::Mul)]
fn mul(self: Value, rhs: f64) -> Value {
    Value::init(
        self.borrow().data * rhs,
        0.0,
        Some(mul_backward),
        vec![self.clone(), Value::new(rhs)],
        Some(Operation::Mul),
    )
}

#[opimps::impl_ops_lprim(ops::Mul)]
fn mul(self: f64, rhs: Value) -> Value {
    Value::init(
        self * rhs.borrow().data,
        0.0,
        Some(mul_backward),
        vec![Value::new(self), rhs.clone()],
        Some(Operation::Mul),
    )
}

fn mul_backward(value: &V) {
    let data0 = value.prev[0].borrow().data;
    let data1 = value.prev[1].borrow().data;
    value.prev[0].borrow_mut().grad += data1 * value.grad;
    value.prev[1].borrow_mut().grad += data0 * value.grad;
}

// POWER and ReLu

impl Value {
    pub fn pow(&self, power: f64) -> Value {
        Value::init(
            self.borrow().data.powf(power),
            0.0,
            Some(|value: &V| {
                let base = value.prev[0].borrow().data;
                let power = value.prev[1].borrow().data;
                value.prev[0].borrow_mut().grad += power * base.powf(power - 1.0) * value.grad;
            }),
            vec![self.clone(), Self::new(power)],
            Some(Operation::Pow),
        )
    }

    pub fn relu(&self) -> Value {
        Value::init(
            self.borrow().data.max(0.0),
            0.0,
            Some(|value: &V| {
                value.prev[0].borrow_mut().grad += if value.data > 0.0 { value.grad } else { 0.0 };
            }),
            vec![self.clone()],
            Some(Operation::ReLU),
        )
    }
}
