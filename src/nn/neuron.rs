use crate::engine::{ActvFn, Op, Value};
use rand::{distributions::Uniform, Rng};
use std::fmt;

pub struct Neuron {
    pub weights: Vec<Value>,
    pub bias: Value,
    pub actv_fn: Option<ActvFn>,
}

impl Neuron {
    pub fn new(nin: u32, nonlin: Option<ActvFn>) -> Neuron {
        let mut rng = rand::thread_rng();
        let range = Uniform::<f64>::new(-1., 1.);

        Neuron {
            weights: (0..nin).map(|_| Value::new(rng.sample(range))).collect(),
            bias: Value::new(0.),
            actv_fn: nonlin,
        }
    }

    pub fn forw(&self, x: &[Value]) -> Value {
        let act = self
            .weights
            .iter()
            .zip(x)
            .map(|(w_i, x_i)| w_i * x_i)
            .sum::<Value>()
            + &self.bias;

        match self.actv_fn {
            Some(ActvFn::ReLU) => act.relu(),
            Some(ActvFn::LeakyReLU) => act.leaky_relu(),
            Some(ActvFn::Tanh) => act.tanh(),
            Some(ActvFn::Sigmoid) => act.sigmoid(),
            None => act,
        }
    }

    pub fn forward(&self, x: &[Vec<Value>]) -> Vec<Value> {
        x.iter().map(|x_i| self.forw(x_i)).collect()
    }

    pub fn parameters(&self) -> Vec<Value> {
        let mut p = self.weights.clone();
        p.push(self.bias.clone());
        p
    }
}

impl fmt::Display for Neuron {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.actv_fn {
            Some(val) => {
                write!(f, "{}({})", Op::ActvFn(val), self.weights.len())
            }
            None => write!(f, "linear({})", self.weights.len()),
        }
    }
}

impl fmt::Debug for Neuron {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debug_struct = f.debug_struct("Neuron");
        debug_struct.field("input", &self.weights.len());
        if let Some(val) = self.actv_fn {
            debug_struct.field("actv_fn", &Op::ActvFn(val));
        }
        debug_struct.finish()
    }
}

impl Neuron {
    pub fn name_params(self) -> Neuron {
        let weights = self
            .weights
            .iter()
            .map(|wi| wi.clone().with_name('w'))
            .collect();
        let bias = self.bias.clone().with_name('b');
        let nonlin = self.actv_fn;

        Neuron {
            weights,
            bias,
            actv_fn: nonlin,
        }
    }

    pub fn name_inputs(&self, x: Vec<Vec<Value>>) -> Vec<Vec<Value>> {
        x.iter()
            .map(|row| row.iter().map(|xij| xij.clone().with_name('X')).collect())
            .collect()
    }
}
