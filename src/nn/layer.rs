use crate::engine::{ActvFn, Value};
use crate::nn::Neuron;
use std::fmt;

pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(nin: u32, nout: u32, nonlin: Option<ActvFn>) -> Layer {
        Layer {
            neurons: (0..nout).map(|_| Neuron::new(nin, nonlin)).collect(),
        }
    }

    /// Forward a single 1d input x through the Layer.
    pub fn forw(&self, x: &[Value]) -> Vec<Value> {
        self.neurons.iter().map(|n| n.forw(x)).collect()
    }

    /// Forward pass of 2d input x through the Layer.
    pub fn forward(&self, x: &[&[Value]]) -> Vec<Vec<Value>> {
        x.iter().map(|xrow| self.forw(xrow)).collect()
    }

    pub fn parameters(&self) -> Vec<Value> {
        self.neurons.iter().flat_map(|n| n.parameters()).collect()
    }
}

impl fmt::Display for Layer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.neurons.first() {
            Some(neuron) => {
                write!(f, "{} → {}", neuron, self.neurons.len())
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
