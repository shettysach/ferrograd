use crate::engine::{Activation, Operation, Value};
use rand::{distributions::Uniform, Rng};
use std::fmt;

pub struct Neuron {
    pub w: Vec<Value>,              // Weights
    pub b: Value,                   // Bias
    pub nonlin: Option<Activation>, // None if linear
}

// Main
impl Neuron {
    // Initialise with uniformly distributed random weights
    pub fn new(nin: i32, nonlin: Option<Activation>) -> Neuron {
        let mut rng = rand::thread_rng();
        let range = Uniform::<f64>::new(-1., 1.);

        Neuron {
            w: (0..nin).map(|_| Value::new(rng.sample(range))).collect(),
            b: Value::new(0.),
            nonlin,
        }
    }

    // Forward pass
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

    // Weights and bias
    pub fn parameters(&self) -> Vec<Value> {
        let mut p = self.w.clone();
        p.insert(0, self.b.clone());
        p
    }
}

// Extra
impl Neuron {
    // Name parameters as 'weight i' and 'bias'
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

    // Name inputs as 'input i'
    pub fn name_inputs(&self, x: Vec<Value>) -> Vec<Value> {
        x.iter()
            .enumerate()
            .map(|(i, xi)| xi.clone().with_name(&format!("input {i}")))
            .collect()
    }
}

// Display trait for printing
impl fmt::Display for Neuron {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let actv_fn = match self.nonlin {
            Some(val) => Operation::AF(val).symbol(),
            None => ' ',
        };

        write!(f, "{}({})", actv_fn, self.w.len())
    }
}
