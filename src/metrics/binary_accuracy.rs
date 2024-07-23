use crate::engine::Value;

pub struct BinaryAccuracy {
    threshold: f64,
}

impl BinaryAccuracy {
    pub fn new(threshold: f64) -> BinaryAccuracy {
        BinaryAccuracy { threshold }
    }

    pub fn compute(
        &self,
        ypred: &Vec<Vec<Value>>,
        ys: &Vec<Vec<Value>>,
    ) -> f64 {
        ypred
            .iter()
            .zip(ys)
            .map(|(ypred_i, ys_i)| {
                ypred_i
                    .iter()
                    .zip(ys_i.iter())
                    .filter(|&(ypred_j, ys_j)| {
                        (ys_j.borrow().data > self.threshold)
                            == (ypred_j.borrow().data > self.threshold)
                    })
                    .count()
            })
            .sum::<usize>() as f64
            / (ypred.len() * ypred[0].len()) as f64
    }
}
