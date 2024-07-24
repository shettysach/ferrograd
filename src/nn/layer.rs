use crate::engine::{Activation, Value};
use crate::nn::Neuron;
use std::fmt;

/// Dense layer of Neurons.
pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    /// Initialise new Layer.
    pub fn new(nin: u32, nout: u32, nonlin: Option<Activation>) -> Layer {
        Layer {
            neurons: (0..nout).map(|_| Neuron::new(nin, nonlin)).collect(),
        }
    }

    /// Forward pass of batch of input xs through the Layer.
    pub fn forward(&self, x: &Vec<Vec<Value>>) -> Vec<Vec<Value>> {
        x.iter()
            .map(|xrow| self.neurons.iter().map(|n| n.forw(xrow)).collect())
            .collect()
    }

    /// Returns vector of weights and biases of the Neurons of the Layer.
    pub fn parameters(&self) -> Vec<Value> {
        self.neurons.iter().flat_map(|n| n.parameters()).collect()
    }
}

impl fmt::Display for Layer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.neurons.iter().next() {
            Some(neuron) => {
                write!(f, "{} * {}", self.neurons.len(), neuron)
            }
            None => write!(f, "Empty"),
        }
    }
}
