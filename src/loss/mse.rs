use crate::engine::Value;

/** Mean Square Error loss (MSE)
To use Root Mean Square Error loss (RMSE), use `mse.loss().pow(0.5)`.*/
#[derive(Debug)]
pub struct MeanSquareErrorLoss {}

impl MeanSquareErrorLoss {
    pub fn new() -> MeanSquareErrorLoss {
        MeanSquareErrorLoss {}
    }

    pub fn loss(&self, ypred: &[Vec<Value>], ytrue: &[Vec<Value>]) -> Value {
        ypred
            .iter()
            .zip(ytrue)
            .map(|(ypred_i, ytrue_i)| {
                ypred_i
                    .iter()
                    .zip(ytrue_i)
                    .map(|(ypred_j, ytrue_j)| (ytrue_j - ypred_j).pow(2.0))
                    .sum::<Value>()
                    / ypred_i.len() as f64
            })
            .sum::<Value>()
            / ypred.len() as f64
    }
}

impl Default for MeanSquareErrorLoss {
    fn default() -> Self {
        Self::new()
    }
}
