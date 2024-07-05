use std::{cell::RefCell, fmt, ops, rc::Rc};
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

    pub fn with_name(self, var_name: &str) -> Value {
        self.borrow_mut().var_name = Some(var_name.to_string());
        self
    }
}

// Removes calling .0 for every borrow
impl ops::Deref for Value {
    type Target = Rc<RefCell<V>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let v = &self.borrow();

        let fmt_name = |var_name: &str| -> String {
            if var_name.is_empty() {
                "".to_string()
            } else {
                format!("← {}", var_name)
                //format!("← \x1B[1m{} \x1B[0m", var_name)
            }
        };

        match (&v.var_name, &v.op) {
            (Some(var_name), Some(op)) => {
                write!(
                    f,
                    "{} data = {:.3}, grad = {:.3} {}",
                    op,
                    v.data,
                    v.grad,
                    fmt_name(var_name)
                )
            }
            (Some(var_name), None) => {
                write!(
                    f,
                    "data = {:.3}, grad = {:.3} {}",
                    v.data,
                    v.grad,
                    fmt_name(var_name)
                )
            }
            (None, _) => {
                write!(f, "{:.3}", v.data)
            }
        }
    }
}

pub enum Operation {
    Add,
    Mul,
    Pow,
    AF(Activation),
}

impl Operation {
    fn symbol(&self) -> char {
        match self {
            Operation::Add => '+',
            Operation::Mul => '*',
            Operation::Pow => '^',
            Operation::AF(Activation::ReLU) => 'R',
            Operation::AF(Activation::LeakyReLU) => 'L',
            Operation::AF(Activation::Tanh) => 't',
            Operation::AF(Activation::Sigmoid) => 'σ',
        }
    }
}

impl fmt::Display for Operation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", &self.symbol())
    }
}

#[derive(Clone, Copy)]
pub enum Activation {
    ReLU,
    LeakyReLU,
    Tanh,
    Sigmoid,
}
