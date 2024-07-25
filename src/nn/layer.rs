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

    /// Forward  single input x through the Layer.
    pub fn forw(&self, x: &Vec<Value>) -> Vec<Value> {
        self.neurons.iter().map(|n| n.forw(x)).collect()
    }

    /// Forward pass of input x through the Layer.
    pub fn forward(&self, x: &Vec<Vec<Value>>) -> Vec<Vec<Value>> {
        x.iter().map(|xrow| self.forw(&xrow)).collect()
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
                write!(f, "{} -> {}", neuron, self.neurons.len())
            }
            None => write!(f, "Empty"),
        }
    }
}

impl fmt::Debug for Layer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debug_struct = f.debug_struct("Layer");
        debug_struct.field("neurons", &self.neurons[0]);
        debug_struct.field("output", &self.neurons.len());
        debug_struct.finish()
    }
}
