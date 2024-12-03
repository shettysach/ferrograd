use crate::engine::value::{Activation, Operation, Value};
use std::{cmp::Ordering, fmt, hash::Hash};

// -- Hash --

impl Hash for Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.borrow().uuid.hash(state);
    }
}

// -- Eq --

impl Eq for Value {}

impl PartialEq for Value {
    fn eq(&self, other: &Value) -> bool {
        self.borrow().uuid == other.borrow().uuid
    }
}

// -- Ord --

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

// -- Debug and Display --

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Value")
            .field("data", &self.borrow().data)
            .field("grad", &self.borrow().grad)
            .field("name", &self.borrow().var_name)
            .field("op", &self.borrow().op)
            .field("prev", &self.borrow().prev)
            .finish()
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let v = &self.borrow();

        let fmt_name = |var_name: &str| -> String {
            if var_name.is_empty() {
                String::new()
            } else {
                format!("← {}", var_name)
            }
        };

        match (&v.var_name, &v.op) {
            (Some(var_name), Some(op)) => {
                write!(
                    f,
                    "{op} data = {:.3}, grad = {:.3} {}",
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
            (None, Some(op)) => {
                write!(f, "{op} data = {:.3}, grad = {:.3}", v.data, v.grad,)
            }
            (None, _) => {
                write!(f, "{:.3}", v.data)
            }
        }
    }
}

impl fmt::Display for Operation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let symbol = match self {
            Operation::Add => "+",
            Operation::Mul => "*",
            Operation::Pow => "^",
            Operation::Ln => "ln",
            Operation::Exp => "exp",
            Operation::AF(Activation::ReLU) => "ReLU",
            Operation::AF(Activation::LeakyReLU) => "LeakyReLU",
            Operation::AF(Activation::Tanh) => "tanh",
            Operation::AF(Activation::Sigmoid) => "σ",
        };
        write!(f, "{}", symbol)
    }
}
