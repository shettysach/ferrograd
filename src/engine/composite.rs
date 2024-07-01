use super::{Operation, Value};
use std::ops;

// NEGATION

#[opimps::impl_uni_ops(ops::Neg)]
fn neg(self: Value) -> Value {
    let res = self * -1.0;
    res.borrow_mut().op = Some(Operation::Neg);
    res
}

// SUBTRACTION

#[opimps::impl_ops(ops::Sub)]
fn sub(self: Value, rhs: Value) -> Value {
    let res = self + (-rhs);
    res.borrow_mut().op = Some(Operation::Sub);
    res
}

#[opimps::impl_ops_rprim(ops::Sub)]
fn sub(self: Value, rhs: f64) -> Value {
    let res = self + (-rhs);
    res.borrow_mut().op = Some(Operation::Sub);
    res
}

#[opimps::impl_ops_lprim(ops::Sub)]
fn sub(self: f64, rhs: Value) -> Value {
    let res = self + (-rhs);
    res.borrow_mut().op = Some(Operation::Sub);
    res
}

// DIVISION

#[opimps::impl_ops(ops::Div)]
fn div(self: Value, rhs: Value) -> Value {
    let res = self * rhs.pow(-1.0);
    res.borrow_mut().op = Some(Operation::Div);
    res
}

#[opimps::impl_ops_rprim(ops::Div)]
fn div(self: Value, rhs: f64) -> Value {
    let res = self * rhs.powf(-1.0);
    res.borrow_mut().op = Some(Operation::Div);
    res
}

#[opimps::impl_ops_lprim(ops::Div)]
fn div(self: f64, rhs: Value) -> Value {
    let res = self * rhs.pow(-1.0);
    res.borrow_mut().op = Some(Operation::Div);
    res
}
