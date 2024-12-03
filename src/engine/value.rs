use std::{cell::RefCell, ops, rc::Rc};
use uuid::Uuid;

/// Scalar with data and gradient.
#[derive(Clone)]
pub struct Value(Rc<RefCell<V>>); // Smart pointer to V

// Struct that holds the data
pub struct V {
    pub data: f64,
    pub grad: f64,
    pub(crate) backward: Option<fn(value: &V)>, // Function pointer to the backward function
    pub(crate) prev: Vec<Value>,                // Children (Operands)
    pub(crate) op: Option<Operation>,           // None if initialisation or constant
    pub(crate) uuid: Uuid,                      // Unique id for easier hashing and eq
    pub(crate) var_name: Option<String>,        // Variable name
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

    /// Initialise new Value from a f64.
    pub fn new(data: f64) -> Value {
        Value::init(data, None, Vec::new(), None, None)
    }

    /// Initialise new 1d vector of Values from a 1d vector of f64.
    pub fn new_1d(data: &[f64]) -> Vec<Value> {
        data.iter().map(|float| Value::new(*float)).collect()
    }

    /// Initialise new 2d vector of Values from a 2d vector of f64.
    pub fn new_2d(data: &[Vec<f64>]) -> Vec<Vec<Value>> {
        data.iter().map(|vec| Value::new_1d(vec)).collect()
    }
}

impl<T: Into<f64>> From<T> for Value {
    fn from(t: T) -> Value {
        Value::new(t.into())
    }
}

impl Value {
    /// Initialise new 1d vector of Values from a 1d vector of type `T`,
    /// where `T` is any value that can be converted to f64.
    pub fn from_1d<T: Into<f64> + Clone>(data: &[T]) -> Vec<Value> {
        data.iter().map(|t| Value::from(t.clone())).collect()
    }

    /// Initialise new 2d vector of Values from a 2d vector of type `T`,
    /// where `T` is any value that can be converted to f64.
    pub fn from_2d<T: Into<f64> + Clone>(data: &[Vec<T>]) -> Vec<Vec<Value>> {
        data.iter().map(|t| Value::from_1d(t)).collect()
    }
}

/// Scalar operations and activation functions.
#[derive(Debug)]
pub enum Operation {
    Add,
    Mul,
    Pow,
    Ln,
    Exp,
    AF(Activation),
}

/// Activation functions.
#[derive(Debug, Clone, Copy)]
pub enum Activation {
    ReLU,
    LeakyReLU,
    Tanh,
    Sigmoid,
}

// --- Extras ---

impl Value {
    /// Assign var_name to the Value
    pub fn with_name(self, var_name: &str) -> Value {
        self.borrow_mut().var_name = Some(var_name.to_string());
        self
    }
}
