use crate::engine::Value;

// Use for binary classification when target values are -1 an 1.
#[derive(Debug)]
pub struct HingeLoss;

impl HingeLoss {
    pub fn new() -> HingeLoss {
        HingeLoss
    }

    pub fn loss(
        &self,
        ypred: &Vec<Vec<Value>>,
        ytrue: &Vec<Vec<Value>>,
    ) -> Value {
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
