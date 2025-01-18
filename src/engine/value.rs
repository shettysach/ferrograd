use std::{cell::RefCell, cmp::Ordering, fmt, hash::Hash, ops::Deref, rc::Rc};
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
    pub(crate) name: Option<char>,
}

#[derive(Debug)]
pub enum Prev {
    Init,
    Unary(Value),
    Binary(Value, Value),
}

#[derive(Debug)]
pub enum Op {
    Add,
    Mul,
    Pow,
    Ln,
    Exp,
    ActvFn(ActvFn),
    Var,
    Const,
}

#[derive(Debug, Clone, Copy)]
pub enum ActvFn {
    ReLU,
    LeakyReLU,
    Tanh,
    Sigmoid,
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
            name,
        })))
    }

    pub fn new(data: f64) -> Value {
        Value::init(data, None, Prev::Init, Op::Var, None)
    }

    pub(crate) fn new_const(data: f64) -> Value {
        Value::init(data, None, Prev::Init, Op::Const, None)
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

    pub fn with_name(self, name: char) -> Value {
        self.borrow_mut().name = Some(name);
        self
    }
}

// val.0.borrow() becomes val.borrow()
impl Deref for Value {
    type Target = Rc<RefCell<V>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: Into<f64>> From<T> for Value {
    fn from(t: T) -> Value {
        Value::new(t.into())
    }
}

impl Hash for Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.borrow().uuid.hash(state);
    }
}

impl Eq for Value {}

impl PartialEq for Value {
    fn eq(&self, other: &Value) -> bool {
        self.borrow().uuid == other.borrow().uuid
    }
}

impl Ord for Value {
    fn cmp(&self, other: &Value) -> Ordering {
        self.borrow()
            .data
            .partial_cmp(&other.borrow().data)
            .expect("Error in comparing f64s")
    }
}

impl PartialOrd for Value {
    fn partial_cmp(&self, other: &Value) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Value")
            .field("data", &self.borrow().data)
            .field("grad", &self.borrow().grad)
            .field("name", &self.borrow().name)
            .field("op", &self.borrow().op)
            .field("prev", &self.borrow().prev)
            .finish()
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let v = &self.borrow();

        match (v.name, &v.op) {
            (Some(name), Op::Var) => {
                write!(f, "data = {:.3}, grad = {:.3} ← {}", v.data, v.grad, name)
            }
            (Some(name), op) => {
                write!(
                    f,
                    "{} data = {:.3}, grad = {:.3} ← {}",
                    op, v.data, v.grad, name
                )
            }
            (None, Op::Const) => {
                write!(f, "{:.3}", v.data)
            }
            (None, op) => {
                write!(f, "{} data = {:.3}, grad = {:.3}", op, v.data, v.grad,)
            }
        }
    }
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let symbol = match self {
            Op::Add => "+",
            Op::Mul => "*",
            Op::Pow => "^",
            Op::Ln => "ln",
            Op::Exp => "exp",
            Op::ActvFn(ActvFn::ReLU) => "ReLU",
            Op::ActvFn(ActvFn::LeakyReLU) => "LeakyReLU",
            Op::ActvFn(ActvFn::Tanh) => "tanh",
            Op::ActvFn(ActvFn::Sigmoid) => "σ",
            _ => "",
        };

        write!(f, "{}", symbol)
    }
}
