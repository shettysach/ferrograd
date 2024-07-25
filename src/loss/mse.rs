use crate::engine::Value;

#[derive(Debug)]
pub struct MeanSquareErrorLoss {}

impl MeanSquareErrorLoss {
    pub fn new() -> MeanSquareErrorLoss {
        MeanSquareErrorLoss {}
    }

    pub fn loss(&self, ypred: &Vec<Vec<Value>>, ys: &Vec<Vec<Value>>) -> Value {
        ypred
            .iter()
            .zip(ys)
            .map(|(ypred_i, ys_i)| {
                ypred_i
                    .iter()
                    .zip(ys_i)
                    .map(|(ypred_j, ys_j)| (ypred_j - ys_j).pow(2.0))
                    .sum::<Value>()
                    / ypred_i.len() as f32
            })
            .sum::<Value>()
            / ypred.len() as f32
    }
}
