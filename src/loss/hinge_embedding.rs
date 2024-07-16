use crate::engine::Value;

pub struct HingeEmbeddingLoss {
    margin: f64,
}

impl HingeEmbeddingLoss {
    pub fn new(margin: f64) -> HingeEmbeddingLoss {
        HingeEmbeddingLoss { margin }
    }
    pub fn loss(&self, ypred: &Vec<Vec<Value>>, ys: &Vec<Vec<Value>>) -> Value {
        let n = ypred.len() as f64;
        ypred
            .iter()
            .zip(ys)
            .map(|(ypred_i, ys_i)| {
                let ni = ypred_i.len() as f64;
                ypred_i
                    .iter()
                    .zip(ys_i.iter())
                    .map(|(ypred_j, ys_j)| {
                        (self.margin - ys_j * ypred_j).relu()
                    })
                    .sum::<Value>()
                    / ni
            })
            .sum::<Value>()
            / n
    }
}
