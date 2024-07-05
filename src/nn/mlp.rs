use crate::engine::{Activation, Value};
use crate::nn::Layer;

pub struct MultiLayerPerceptron {
    layers: Vec<Layer>,
}

impl MultiLayerPerceptron {
    pub fn new(
        nin: i32,
        mut nouts: Vec<i32>,
        actv_fn: Activation,
    ) -> MultiLayerPerceptron {
        nouts.insert(0, nin);
        let n = nouts.len() - 1;

        let layers = (0..n)
            .map(|i| {
                let nin = nouts[i];
                let nout = nouts[i + 1];
                let nonlin = if i == n - 1 { None } else { Some(actv_fn) };

                Layer::new(nin, nout, nonlin)
            })
            .collect();

        MultiLayerPerceptron { layers }
    }

    pub fn forward(&self, x: Vec<Value>) -> Vec<Value> {
        self.layers.iter().fold(x, |x, layer| layer.forward(&x))
    }

    pub fn parameters(&self) -> Vec<Value> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }

    pub fn zero_grad(&self) {
        for p in self.parameters() {
            p.borrow_mut().grad = 0.0;
        }
    }
}
