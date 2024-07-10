use crate::engine::{Activation, Value};
use crate::nn::Layer;

pub struct MultiLayerPerceptron {
    layers: Vec<Layer>,
}

impl MultiLayerPerceptron {
    pub fn new(
        nin: i32,            // Number of inputs
        mut nouts: Vec<i32>, // Specifies number of neurons in each layer
        actv_fn: Activation,
    ) -> MultiLayerPerceptron {
        // Insert number of inputs into 0th index
        nouts.insert(0, nin);
        // Length-1 because final element is number of outputs
        let n = nouts.len() - 1;

        let layers = (0..n)
            .map(|i| {
                let nin = nouts[i];
                let nout = nouts[i + 1];
                let nonlin = if i == n - 1 { None } else { Some(actv_fn) };
                // Only the last layer has the activation function

                Layer::new(nin, nout, nonlin)
            })
            .collect();

        MultiLayerPerceptron { layers }
    }

    // Forwarding from first layer to the last
    pub fn forward(&self, x: Vec<Value>) -> Vec<Value> {
        self.layers.iter().fold(x, |x, layer| layer.forward(&x))
    }

    // Weights and biases of all the neurons of the perceptron
    pub fn parameters(&self) -> Vec<Value> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }
}
