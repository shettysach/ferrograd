use crate::engine::{ActvFn, Value};
use crate::nn::Layer;
use std::fmt;

pub struct MultiLayerPerceptron {
    layers: Vec<Layer>,
}

impl MultiLayerPerceptron {
    pub fn new(nin: u32, mut nouts: Vec<u32>, actv_fn: ActvFn) -> MultiLayerPerceptron {
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
    pub fn forw(&self, x: &[Value]) -> Vec<Value> {
        self.layers
            .iter()
            .fold(x.to_vec(), |x, layer| layer.forw(&x))
    }

    /// Forward pass of 2d input x through the MLP.
    pub fn forward(&self, x: &[Vec<Value>]) -> Vec<Vec<Value>> {
        x.iter().map(|xrow| self.forw(xrow)).collect()
    }

    pub fn parameters(&self) -> Vec<Value> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }
}

impl fmt::Display for MultiLayerPerceptron {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "MLP:").unwrap();
        for (i, layer) in self.layers.iter().enumerate() {
            writeln!(f, "  layer {}: [ {} ]", i, layer).unwrap();
        }
        Ok(())
    }
}

impl fmt::Debug for MultiLayerPerceptron {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debug_struct = f.debug_struct("MultiLayerPerceptron");
        debug_struct.field("parameters", &self.parameters().len());
        for (i, layer) in self.layers.iter().enumerate() {
            debug_struct.field(&format!("layer {}", i), layer);
        }
        debug_struct.finish()
    }
}
