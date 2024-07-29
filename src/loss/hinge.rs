use crate::engine::Value;

/** Hinge loss
For binary classification, when targets are 0 and 1.*/
#[derive(Debug)]
pub struct HingeLoss;

impl HingeLoss {
    pub fn new() -> HingeLoss {
        HingeLoss
    }

    pub fn loss(&self, ypred: &[Vec<Value>], ytrue: &[Vec<Value>]) -> Value {
        ypred
            .iter()
            .zip(ytrue.iter())
            .map(|(ypred_i, ytrue_i)| {
                ypred_i
                    .iter()
                    .zip(ytrue_i)
                    .map(|(ypred_j, ytrue)| (1.0 - ytrue * ypred_j).relu())
                    .sum::<Value>()
            })
            .sum::<Value>()
            / (ypred.len() * ypred[0].len()) as f64
    }
}

impl Default for HingeLoss {
    fn default() -> Self {
        Self::new()
    }
}
