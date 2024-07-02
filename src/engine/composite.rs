use super::Value;
use std::ops;

// NEGATION

#[opimps::impl_uni_ops(ops::Neg)]
fn neg(self: Value) -> Value {
    self * -1.0
}

// SUBTRACTION

#[opimps::impl_ops(ops::Sub)]
fn sub(self: Value, rhs: Value) -> Value {
    self + (-rhs)
}

#[opimps::impl_ops_rprim(ops::Sub)]
fn sub(self: Value, rhs: f64) -> Value {
    self + (-rhs)
}

#[opimps::impl_ops_lprim(ops::Sub)]
fn sub(self: f64, rhs: Value) -> Value {
    self + (-rhs)
}

// DIVISION

#[opimps::impl_ops(ops::Div)]
fn div(self: Value, rhs: Value) -> Value {
    self * rhs.pow(-1.0)
}

#[opimps::impl_ops_rprim(ops::Div)]
fn div(self: Value, rhs: f64) -> Value {
    self * rhs.powf(-1.0)
}

#[opimps::impl_ops_lprim(ops::Div)]
fn div(self: f64, rhs: Value) -> Value {
    self * rhs.pow(-1.0)
}
