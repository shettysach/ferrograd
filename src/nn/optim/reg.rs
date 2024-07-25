use crate::engine::Value;

pub fn l1_regularization(alpha: f64, params: Vec<Value>) -> Value {
    alpha * params.into_iter().sum::<Value>()
}

pub fn l2_regularization(alpha: f64, params: Vec<Value>) -> Value {
    alpha * params.iter().map(|p| p * p).sum::<Value>()
}
