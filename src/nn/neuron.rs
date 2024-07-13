use crate::engine::{Activation, Operation, Value};
use rand::{distributions::Uniform, Rng};
use std::fmt;

/// Neuron with weights and a bias.
pub struct Neuron {
    pub w: Vec<Value>,              // Weights
    pub b: Value,                   // Bias
    pub nonlin: Option<Activation>, // None if linear
}

impl Neuron {
    /// Initialise new Neuron with uniformly distributed random weights and zero bias.
    pub fn new(nin: u32, nonlin: Option<Activation>) -> Neuron {
        let mut rng = rand::thread_rng();
        let range = Uniform::<f64>::new(-1., 1.);

        Neuron {
            w: (0..nin).map(|_| Value::new(rng.sample(range))).collect(),
            b: Value::new(0.),
            nonlin,
        }
    }

    /// Forward pass of input x through the Neuron.
    pub fn forward(&self, x: &Vec<Value>) -> Value {
        let act = self.w.iter().zip(x).map(|(wi, xi)| wi * xi).sum::<Value>()
            + &self.b;

        match self.nonlin {
            Some(Activation::ReLU) => act.relu(),
            Some(Activation::LeakyReLU) => act.leaky_relu(),
            Some(Activation::Tanh) => act.tanh(),
            Some(Activation::Sigmoid) => act.sigmoid(),
            None => act,
        }
    }

    /// Returns vector of bias and weights.
    pub fn parameters(&self) -> Vec<Value> {
        let mut p = self.w.clone();
        p.insert(0, self.b.clone());
        p
    }
}

// Display trait for printing
impl fmt::Display for Neuron {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.nonlin {
            Some(val) => write!(f, "{}({})", Operation::AF(val), self.w.len()),
            None => write!(f, "({})", self.w.len()),
        }
    }
}

// -- Extras --

impl Neuron {
    /// Assign var_names for parameters as 'weight i' and 'bias'.
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

    /// Assign var_names for inputs as 'input i'.
    pub fn name_inputs(&self, x: Vec<Value>) -> Vec<Value> {
        x.iter()
            .enumerate()
            .map(|(i, xi)| xi.clone().with_name(&format!("input {i}")))
            .collect()
    }
}
