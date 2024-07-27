use crate::engine::Value;

/** Cross-Entropy loss
For multiclass and multilabel classification.*/
#[derive(Debug)]
pub struct CrossEntropyLoss;

impl CrossEntropyLoss {
    pub fn new() -> CrossEntropyLoss {
        CrossEntropyLoss
    }

    pub fn loss(
        &self,
        ypred: &Vec<Vec<Value>>,
        ytrue: &Vec<Vec<Value>>,
    ) -> Value {
        -ypred
            .iter()
            .zip(ytrue)
            .map(|(ypred_i, ytrue_i)| {
                ypred_i
                    .iter()
                    .zip(ytrue_i)
                    .map(|(ypred_j, ytrue_j)| ytrue_j * ypred_j.ln())
                    .sum::<Value>()
            })
            .sum::<Value>()
            / (ypred.len() * ypred[0].len()) as f64
    }
}
