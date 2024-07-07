use crate::{
    engine::{Activation, Operation, Value},
    nn::Neuron,
};
use std::fmt;
use termtree::Tree;

impl Value {
    pub fn with_name(self, var_name: &str) -> Value {
        self.borrow_mut().var_name = Some(var_name.to_string());
        self
    }

    pub fn tree(&self) -> Tree<Value> {
        let mut root = Tree::new(self.clone());
        if self.borrow().op.is_some() {
            self.borrow().prev.iter().for_each(|p| {
                root.push(p.tree());
            })
        }
        root
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

impl Neuron {
    pub fn name_params(self) -> Neuron {
        let w = self
            .w
            .iter()
            .enumerate()
            .map(|(i, wi)| wi.clone().with_name(&format!("weight {i}")))
            .collect();
        let b = self.b.clone().with_name("bias");
        let nonlin = self.nonlin;
        Neuron { w, b, nonlin }
    }

    pub fn name_inputs(&self, x: Vec<Value>) -> Vec<Value> {
        x.iter()
            .enumerate()
            .map(|(i, xi)| xi.clone().with_name(&format!("input {i}")))
            .collect()
    }
}

impl fmt::Display for Neuron {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let actv_fn = match self.nonlin {
            Some(val) => Operation::AF(val).symbol(),
            None => ' ',
        };

        write!(f, "{}({})", actv_fn, self.w.len())
    }
}
