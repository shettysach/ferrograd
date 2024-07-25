use std::{cell::RefCell, ops, rc::Rc};
use uuid::Uuid;

/// Scalar with data and gradient.
#[derive(Clone)]
pub struct Value(Rc<RefCell<V>>); // Smart pointer to V

// Struct that holds the data
pub struct V {
    pub data: f32,
    pub grad: f32,
    pub _backward: Option<fn(value: &V)>, // Function pointer to the backward function
    pub _prev: Vec<Value>,                // Children (Operands)
    pub _op: Option<Operation>,           // None if initialisation or constant
    pub _uuid: Uuid,                      // Unique id for easier hashing and eq
    pub _var_name: Option<String>,        // None if constant
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
        data: f32,
        backward: Option<fn(value: &V)>,
        prev: Vec<Value>,
        op: Option<Operation>,
        var_name: Option<String>,
    ) -> Value {
        Value(Rc::new(RefCell::new(V {
            data,
            grad: 0.0,
            _backward: backward,
            _prev: prev,
            _op: op,
            _uuid: Uuid::new_v4(),
            _var_name: var_name,
        })))
    }

    /// Initialise new Value.
    pub fn new(data: f32) -> Value {
        Value::init(data, None, Vec::new(), None, Some(String::new()))
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
        self.borrow_mut()._var_name = Some(var_name.to_string());
        self
    }
}
