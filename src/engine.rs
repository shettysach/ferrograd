use std::{cell::RefCell, fmt, ops, rc::Rc};
use uuid::Uuid;

mod activation;
mod backprop;
mod composite_ops;
mod primitive_ops;

/// Scalar with data and gradient.
#[derive(Clone)]
pub struct Value(Rc<RefCell<V>>); // Smart pointer to V

// Struct that holds the data
pub struct V {
    pub data: f64,
    pub grad: f64,
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
        data: f64,
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
    pub fn new(data: f64) -> Value {
        Value::init(data, None, Vec::new(), None, Some(String::new()))
    }
}

// Display trait for printing
impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let v = &self.borrow();

        let fmt_name = |var_name: &str| -> String {
            if var_name.is_empty() {
                "".to_string()
            } else {
                format!("← {}", var_name)
            }
        };

        match (&v._var_name, &v._op) {
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

/// Primitive scalar operations and activation functions.
pub enum Operation {
    Add,
    Mul,
    Pow,
    AF(Activation),
}

/// Activation functions.
#[derive(Clone, Copy)]
pub enum Activation {
    ReLU,
    LeakyReLU,
    Tanh,
    Sigmoid,
}

// Display trait for printing
impl fmt::Display for Operation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let symbol = match self {
            Operation::Add => '+',
            Operation::Mul => '*',
            Operation::Pow => '^',
            Operation::AF(Activation::ReLU) => 'R',
            Operation::AF(Activation::LeakyReLU) => 'L',
            Operation::AF(Activation::Tanh) => 't',
            Operation::AF(Activation::Sigmoid) => 'σ',
        };
        write!(f, "{}", symbol)
    }
}

// --- Extras ---

impl Value {
    /// Assign var_name to the Value
    pub fn with_name(self, var_name: &str) -> Value {
        self.borrow_mut()._var_name = Some(var_name.to_string());
        self
    }
}
