use crate::engine::Value;

pub struct BinaryCrossEntropyLoss {}

impl BinaryCrossEntropyLoss {
    pub fn new() -> BinaryCrossEntropyLoss {
        BinaryCrossEntropyLoss {}
    }

    pub fn loss(&self, ypred: &Vec<Vec<Value>>, ys: &Vec<Vec<Value>>) -> Value {
        let epsilon = 1e-7;
        ypred
            .iter()
            .zip(ys)
            .map(|(ypred_i, ys_i)| {
                ypred_i
                    .iter()
                    .zip(ys_i)
                    .map(|(ypred_j, ys_j)| {
                        let data_clamped =
                            ypred_j.borrow().data.clamp(epsilon, 1.0 - epsilon);
                        let yp = Value::new(data_clamped).sigmoid().ln();

                        -((ys_j * &yp) + (1.0 - ys_j) * (1.0 - &yp))
                    })
                    .sum::<Value>()
                    / ypred_i.len() as f64
            })
            .sum::<Value>()
            / ypred.len() as f64
    }
}
