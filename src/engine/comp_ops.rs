use crate::engine::value::Value;
use std::{iter::Sum, ops};

// Negation

#[opimps::impl_uni_ops(ops::Neg)]
fn neg(self: Value) -> Value {
    self * -1.0
}

// Subtraction

#[opimps::impl_ops(ops::Sub)]
fn sub(self: Value, rhs: Value) -> Value {
    self + (-rhs)
}

#[opimps::impl_ops_rprim(ops::Sub)]
fn sub(self: Value, rhs: f32) -> Value {
    self + (-rhs)
}

#[opimps::impl_ops_lprim(ops::Sub)]
fn sub(self: f32, rhs: Value) -> Value {
    self + (-rhs)
}

// Division

#[opimps::impl_ops(ops::Div)]
fn div(self: Value, rhs: Value) -> Value {
    self * rhs.pow(-1.0)
}

#[opimps::impl_ops_rprim(ops::Div)]
fn div(self: Value, rhs: f32) -> Value {
    self * rhs.powf(-1.0)
}

#[opimps::impl_ops_lprim(ops::Div)]
fn div(self: f32, rhs: Value) -> Value {
    self * rhs.pow(-1.0)
}

// Assigns

#[opimps::impl_ops_assign(ops::AddAssign)]
fn add_assign(self: Value, rhs: Value) {
    let name = self.borrow()._var_name.clone();
    *self = &*self + rhs;
    self.borrow_mut()._var_name = name;
}

#[opimps::impl_ops_assign(ops::MulAssign)]
fn mul_assign(self: Value, rhs: Value) {
    let name = self.borrow()._var_name.clone();
    *self = &*self * rhs;
    self.borrow_mut()._var_name = name;
}

#[opimps::impl_ops_assign(ops::SubAssign)]
fn sub_assign(self: Value, rhs: Value) {
    let name = self.borrow()._var_name.clone();
    *self = &*self - rhs;
    self.borrow_mut()._var_name = name;
}

#[opimps::impl_ops_assign(ops::DivAssign)]
fn div_assign(self: Value, rhs: Value) {
    let name = self.borrow()._var_name.clone();
    *self = &*self / rhs;
    self.borrow_mut()._var_name = name;
}

// Sum

impl Sum for Value {
    fn sum<I>(iter: I) -> Value
    where
        I: Iterator<Item = Self>,
    {
        let mut iter = iter;
        match iter.next() {
            Some(first) => iter.fold(first, |acc, val| acc + val),
            None => Value::new(0.0),
        }
    }
}
