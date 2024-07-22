use crate::engine::Value;

pub struct HingeLoss {
    margin: f64,
}

impl HingeLoss {
    pub fn new(margin: f64) -> HingeLoss {
        HingeLoss { margin }
    }

    pub fn loss(&self, ypred: &Vec<Vec<Value>>, ys: &Vec<Vec<Value>>) -> Value {
        ypred
            .iter()
            .zip(ys)
            .map(|(ypred_i, ys_i)| {
                ypred_i
                    .iter()
                    .zip(ys_i.iter())
                    .map(|(ypred_j, ys_j)| {
                        (self.margin - ys_j * ypred_j).relu()
                    })
                    .sum::<Value>()
                    / ypred_i.len() as f64
            })
            .sum::<Value>()
            / ypred.len() as f64
    }
}
