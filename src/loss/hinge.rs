use crate::engine::Value;

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
            .zip(ytrue)
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
