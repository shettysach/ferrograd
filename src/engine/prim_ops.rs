use crate::engine::value::{Op, Prev, Value, V};
use std::ops;

// Addition

#[opimps::impl_ops(ops::Add)]
fn add(self: Value, rhs: Value) -> Value {
    Value::init(
        self.borrow().data + rhs.borrow().data,
        Some(add_backward),
        Prev::Binary(self.clone(), rhs.clone()),
        Op::Add,
        None,
    )
}

#[opimps::impl_ops_rprim(ops::Add)]
fn add(self: Value, rhs: f64) -> Value {
    Value::init(
        self.borrow().data + rhs,
        Some(add_backward_lhs),
        Prev::Binary(self.clone(), Value::new_const(rhs)),
        Op::Add,
        None,
    )
}

#[opimps::impl_ops_lprim(ops::Add)]
fn add(self: f64, rhs: Value) -> Value {
    Value::init(
        self + rhs.borrow().data,
        Some(add_backward_rhs),
        Prev::Binary(Value::new_const(self), rhs.clone()),
        Op::Add,
        None,
    )
}

fn add_backward(value: &V) {
    if let Prev::Binary(l, r) = &value.prev {
        l.borrow_mut().grad += value.grad;
        r.borrow_mut().grad += value.grad;
    }
}

fn add_backward_lhs(value: &V) {
    if let Prev::Binary(l, _) = &value.prev {
        l.borrow_mut().grad += value.grad;
    }
}

fn add_backward_rhs(value: &V) {
    if let Prev::Binary(_, r) = &value.prev {
        r.borrow_mut().grad += value.grad;
    }
}

// Multiplication

#[opimps::impl_ops(ops::Mul)]
fn mul(self: Value, rhs: Value) -> Value {
    Value::init(
        self.borrow().data * rhs.borrow().data,
        Some(mul_backward),
        Prev::Binary(self.clone(), rhs.clone()),
        Op::Mul,
        None,
    )
}

#[opimps::impl_ops_rprim(ops::Mul)]
fn mul(self: Value, rhs: f64) -> Value {
    Value::init(
        self.borrow().data * rhs,
        Some(mul_backward_lhs),
        Prev::Binary(self.clone(), Value::new_const(rhs)),
        Op::Mul,
        None,
    )
}

#[opimps::impl_ops_lprim(ops::Mul)]
fn mul(self: f64, rhs: Value) -> Value {
    Value::init(
        self * rhs.borrow().data,
        Some(mul_backward_rhs),
        Prev::Binary(Value::new_const(self), rhs.clone()),
        Op::Mul,
        None,
    )
}

fn mul_backward(value: &V) {
    if let Prev::Binary(l, r) = &value.prev {
        let l_data = l.borrow().data;
        let r_data = r.borrow().data;
        l.borrow_mut().grad += r_data * value.grad;
        r.borrow_mut().grad += l_data * value.grad;
    }
}

fn mul_backward_lhs(value: &V) {
    if let Prev::Binary(l, r) = &value.prev {
        let r_data = r.borrow().data;
        l.borrow_mut().grad += r_data * value.grad;
    }
}

fn mul_backward_rhs(value: &V) {
    if let Prev::Binary(l, r) = &value.prev {
        let l_data = l.borrow().data;
        r.borrow_mut().grad += l_data * value.grad;
    }
}

// Power, Ln and Exp

impl Value {
    pub fn pow(&self, power: f64) -> Value {
        Value::init(
            self.borrow().data.powf(power),
            Some(|value: &V| {
                if let Prev::Binary(a, b) = &value.prev {
                    let base = a.borrow().data;
                    let power = b.borrow().data;
                    a.borrow_mut().grad += power * base.powf(power - 1.0) * value.grad;
                }
            }),
            Prev::Binary(self.clone(), Value::new_const(power)),
            Op::Pow,
            None,
        )
    }

    pub fn ln(&self) -> Value {
        Value::init(
            self.borrow().data.ln(),
            Some(|value: &V| {
                if let Prev::Unary(a) = &value.prev {
                    let mut prev = a.borrow_mut();
                    prev.grad += value.grad / prev.data;
                }
            }),
            Prev::Unary(self.clone()),
            Op::Ln,
            None,
        )
    }

    pub fn exp(&self) -> Value {
        Value::init(
            self.borrow().data.exp(),
            Some(|value: &V| {
                if let Prev::Unary(a) = &value.prev {
                    a.borrow_mut().grad += value.data * value.grad;
                }
            }),
            Prev::Unary(self.clone()),
            Op::Exp,
            None,
        )
    }
}
