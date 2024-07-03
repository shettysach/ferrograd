use crate::engine::{Activation, Value};
use rand::{distributions::Uniform, Rng};

pub struct Neuron {
    w: Vec<Value>,
    b: Value,
    nonlin: Option<Activation>,
}

impl Neuron {
    pub fn new(nin: i32, nonlin: Option<Activation>) -> Neuron {
        let mut rng = rand::thread_rng();
        let range = Uniform::<f64>::new(-1., 1.);

        Neuron {
            w: (0..nin).map(|_| Value::new(rng.sample(range))).collect(),
            b: Value::new(0.),
            nonlin,
        }
    }

    pub fn forward(&self, x: &Vec<Value>) -> Value {
        let act = self.w.iter().zip(x).map(|(wi, xi)| wi * xi).sum::<Value>()
            + &self.b;

        match self.nonlin {
            Some(Activation::ReLU) => act.relu(),
            Some(Activation::Tanh) => act.tanh(),
            Some(Activation::Sigmoid) => act.sigmoid(),
            None => act,
        }
    }

    pub fn parameters(&self) -> Vec<Value> {
        let mut p = self.w.clone();
        p.insert(0, self.b.clone());
        p
    }

    pub fn with_names(self) -> Neuron {
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
}
