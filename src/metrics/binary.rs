use crate::engine::Value;

/// Binary Accuracy
#[derive(Debug)]
pub struct BinaryAccuracy {
    threshold: f64,
}

impl BinaryAccuracy {
    /// Initialise new accuracy metric.
    pub fn new(threshold: f64) -> BinaryAccuracy {
        BinaryAccuracy { threshold }
    }

    /// Compute accuracy.
    pub fn compute(&self, ypred: &[Vec<Value>], ytrue: &[Vec<Value>]) -> f64 {
        ypred
            .iter()
            .zip(ytrue)
            .map(|(ypred_i, ytrue_i)| {
                ypred_i
                    .iter()
                    .zip(ytrue_i.iter())
                    .filter(|&(ypred_j, ytrue_j)| {
                        (ytrue_j.borrow().data > self.threshold)
                            == (ypred_j.borrow().data > self.threshold)
                    })
                    .count()
            })
            .sum::<usize>() as f64
            / (ypred.len() * ypred[0].len()) as f64
    }
}
