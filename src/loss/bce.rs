use crate::engine::Value;

// Use for binary classification when target values are 0 an 1.
pub struct BinaryCrossEntropyLoss;

impl BinaryCrossEntropyLoss {
    pub fn new() -> BinaryCrossEntropyLoss {
        BinaryCrossEntropyLoss
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
                    .map(|(ypred_j, ytrue_j)| {
                        let yp_sigmoid = ypred_j.sigmoid();

                        -((ytrue_j * yp_sigmoid.ln())
                            + (1.0 - ytrue_j) * (1.0 - yp_sigmoid).ln())
                    })
                    .sum::<Value>()
            })
            .sum::<Value>()
            / (ypred.len() * ypred[0].len()) as f64
    }
}
