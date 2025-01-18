use std::{cell::RefCell, ops, rc::Rc};
use uuid::Uuid;

#[derive(Clone)]
pub struct Value(Rc<RefCell<V>>);

pub struct V {
    pub data: f64,
    pub grad: f64,
    pub(crate) backward: Option<fn(value: &V)>,
    pub(crate) prev: Prev,
    pub(crate) op: Op,
    pub(crate) uuid: Uuid,
    pub(crate) var_name: Option<char>,
}

#[derive(Debug)]
pub enum Prev {
    Init,
    Unary(Value),
    Binary(Value, Value),
}

// val.0.borrow() becomes val.borrow()
impl ops::Deref for Value {
    type Target = Rc<RefCell<V>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Value {
    pub fn init(
        data: f64,
        backward: Option<fn(value: &V)>,
        prev: Prev,
        op: Op,
        name: Option<char>,
    ) -> Value {
        Value(Rc::new(RefCell::new(V {
            data,
            grad: 0.0,
            backward,
            prev,
            op,
            uuid: Uuid::new_v4(),
            var_name: name,
        })))
    }

    pub fn new(data: f64) -> Value {
        Value::init(data, None, Prev::Init, Op::Init, None)
    }

    pub fn new_1d(data: &[f64]) -> Vec<Value> {
        data.iter().map(|float| Value::new(*float)).collect()
    }

    pub fn new_2d(data: &[&[f64]]) -> Vec<Vec<Value>> {
        data.iter().map(|vec| Value::new_1d(vec)).collect()
    }

    pub fn from_1d<T: Into<f64> + Clone>(data: &[T]) -> Vec<Value> {
        data.iter().map(|t| Value::from(t.clone())).collect()
    }

    pub fn from_2d<T: Into<f64> + Clone>(data: &[&[T]]) -> Vec<Vec<Value>> {
        data.iter().map(|t| Value::from_1d(t)).collect()
    }

    pub fn new_with_name(data: f64, name: char) -> Value {
        Value::init(data, None, Prev::Init, Op::Init, Some(name))
    }

    pub fn with_name(self, name: char) -> Value {
        self.borrow_mut().var_name = Some(name);
        self
    }
}

impl<T: Into<f64>> From<T> for Value {
    fn from(t: T) -> Value {
        Value::new(t.into())
    }
}

#[derive(Debug)]
pub enum Op {
    Init,
    Add,
    Mul,
    Pow,
    Ln,
    Exp,
    ActvFn(ActvFn),
}

#[derive(Debug, Clone, Copy)]
pub enum ActvFn {
    ReLU,
    LeakyReLU,
    Tanh,
    Sigmoid,
}
