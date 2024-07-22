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
        let ypred = ypred.get(0).expect("Empty predictions");
        let ys = ys.get(0).expect("Empty targets");

        ypred
            .iter()
            .zip(ys)
            .filter(|&(ypred_i, ys_i)| {
                (ys_i.borrow().data > self.threshold)
                    == (ypred_i.borrow().data > self.threshold)
            })
            .count() as f64
            / ypred.len() as f64
    }
}
