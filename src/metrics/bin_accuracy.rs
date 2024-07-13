use crate::engine::Value;

/// Frequency of input matching target.
pub struct BinaryAccuracy {
    threshold: f64,
}

impl BinaryAccuracy {
    /// Initialise new BinaryAccuracy
    pub fn new(threshold: f64) -> BinaryAccuracy {
        BinaryAccuracy { threshold }
    }

    // Compute accuracy
    pub fn compute(
        &self,
        ypred: &Vec<Vec<Value>>,
        ys: &Vec<Vec<Value>>,
    ) -> f64 {
        let n = ypred.len() as f64;

        ypred
            .iter()
            .zip(ys)
            .map(|(ypred_i, ys_i)| {
                let ni = ypred_i.len();

                ypred_i
                    .iter()
                    .zip(ys_i.iter())
                    .filter(|&(ypred_j, ys_j)| {
                        (ys_j.borrow().data > self.threshold)
                            == (ypred_j.borrow().data > self.threshold)
                    })
                    .count()
                    / ni
            })
            .sum::<usize>() as f64
            / n
    }
}
