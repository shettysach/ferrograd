use crate::engine::value::{Activation, Operation, Value};
use std::{cmp::Ordering, fmt, hash::Hash};

impl Hash for Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.borrow()._uuid.hash(state);
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Value) -> bool {
        self.borrow()._uuid == other.borrow()._uuid
    }
}

impl Eq for Value {}

impl PartialOrd for Value {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.borrow().data.partial_cmp(&other.borrow().data)
    }
}

impl Ord for Value {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).expect("Error in comparing floats")
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Value")
            .field("data", &self.borrow().data)
            .field("grad", &self.borrow().grad)
            .field("name", &self.borrow()._var_name)
            .field("op", &self.borrow()._op)
            .field("prev", &self.borrow()._prev)
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
