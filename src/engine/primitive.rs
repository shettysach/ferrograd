use super::{Operation, Value, V};
use std::ops;

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

    fn new_const(data: f64) -> Value {
        Value::init(data, None, Vec::new(), None, None)
    }
}
