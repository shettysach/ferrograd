use std::{cell::RefCell, ops, rc::Rc};
use uuid::Uuid;

mod activation;
mod backpropagation;
mod composite;
mod primitive;

#[derive(Clone)]
pub struct Value(Rc<RefCell<V>>);

pub struct V {
    pub data: f64,
    pub grad: f64,
    pub backward: Option<fn(value: &V)>,
    pub prev: Vec<Value>,
    pub op: Option<Operation>,
    pub uuid: Uuid,
    pub var_name: Option<String>,
}

impl Value {
    pub fn init(
        data: f64,
        backward: Option<fn(value: &V)>,
        prev: Vec<Value>,
        op: Option<Operation>,
        var_name: Option<String>,
    ) -> Value {
        Value(Rc::new(RefCell::new(V {
            data,
            grad: 0.0,
            backward,
            prev,
            op,
            uuid: Uuid::new_v4(),
            var_name,
        })))
    }

    pub fn new(data: f64) -> Value {
        Value::init(data, None, Vec::new(), None, Some(String::new()))
    }
}

impl ops::Deref for Value {
    type Target = Rc<RefCell<V>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub enum Operation {
    Add,
    Mul,
    Pow,
    AF(Activation),
}

#[derive(Clone, Copy)]
pub enum Activation {
    ReLU,
    LeakyReLU,
    Tanh,
    Sigmoid,
}
