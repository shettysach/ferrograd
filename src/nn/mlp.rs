use crate::engine::{Activation, Value};
use crate::nn::Layer;
use std::fmt;

/// Artificial neural network with dense layers and a non-linear activation function.
pub struct MultiLayerPerceptron {
    layers: Vec<Layer>,
}

impl MultiLayerPerceptron {
    /// Initialise new MLP.
    pub fn new(
        nin: u32,
        mut nouts: Vec<u32>,
        actv_fn: Activation,
    ) -> MultiLayerPerceptron {
        nouts.insert(0, nin);
        let n = nouts.len() - 1;

        let layers = (0..n)
            .map(|i| {
                let nin = nouts[i];
                let nout = nouts[i + 1];
                let nonlin = if i == n - 1 { None } else { Some(actv_fn) }; // Last layer is linear

                Layer::new(nin, nout, nonlin)
            })
            .collect();

        MultiLayerPerceptron { layers }
    }

    /// Forward pass of a single 1d input x through the MLP.
    pub fn forw(&self, x: &Vec<Value>) -> Vec<Value> {
        self.layers
            .iter()
            .fold(x.clone(), |x, layer| layer.forw(&x))
    }

    /// Forward pass of 2d input x through the MLP.
    pub fn forward(&self, x: &Vec<Vec<Value>>) -> Vec<Vec<Value>> {
        x.iter().map(|xrow| self.forw(&xrow)).collect()
    }

    /// Returns 1d vec of weights and biases of all the neurons of the perceptron.
    pub fn parameters(&self) -> Vec<Value> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }
}

impl fmt::Display for MultiLayerPerceptron {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let _ = self.layers.iter().enumerate().for_each(|(i, layer)| {
            write!(f, "MLP:").unwrap();
            write!(f, "  layer {}: [ {} ]\n", i, layer).unwrap();
        });
        Ok(())
    }
}

impl fmt::Debug for MultiLayerPerceptron {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debug_struct = f.debug_struct("MultiLayerPerceptron");
        debug_struct.field("parameters", &self.parameters().len());
        self.layers.iter().enumerate().for_each(|(i, layer)| {
            debug_struct.field(&format!("layer {}", i), layer);
        });
        debug_struct.finish()
    }
}
