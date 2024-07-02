use std::{cell::RefCell, fmt, ops, rc::Rc};
use uuid::Uuid;

mod backprop;
mod composite;
mod primitive;

// Smart pointer to V
// Allows multiple owners to share mutable data
// by enforcing borrow rules at runtime
#[derive(Clone)]
pub struct Value(Rc<RefCell<V>>);

pub struct V {
    pub data: f64,
    pub grad: f64,
    pub backward: Option<fn(value: &V)>, // Function pointer
    pub prev: Vec<Value>,
    pub op: Option<Operation>,
    pub uuid: Uuid,
}

impl Value {
    pub fn init(
        data: f64,
        grad: f64,
        backward: Option<fn(value: &V)>,
        prev: Vec<Value>,
        op: Option<Operation>,
    ) -> Self {
        Value(Rc::new(RefCell::new(V {
            data,
            grad,
            backward,
            prev,
            op,
            uuid: Uuid::new_v4(),
        })))
    }

    pub fn new(data: f64) -> Self {
        Self::init(data, 0.0, None, Vec::new(), None)
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

        if let Some(op) = &v.op {
            write!(f, "{} d = {} • g = {}", op, v.data, v.grad,)
        } else {
            write!(f, "d = {} • g = {}", v.data, v.grad,)
        }
    }
}

pub enum Operation {
    Add,
    Mul,
    Pow,
    ReLU,
}

impl Operation {
    fn symbol(&self) -> char {
        match self {
            Self::Add => '+',
            Self::Mul => '*',
            Self::Pow => '^',
            Self::ReLU => 'r',
        }
    }
}

impl fmt::Display for Operation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", &self.symbol())
    }
}
