use crate::engine::Value;

pub fn softmax(y: &[Vec<Value>]) -> Vec<Vec<Value>> {
    y.iter()
        .map(|y_i| {
            let exp_yi = y_i.iter().map(|y_ij| y_ij.exp());
            let exp_sum = exp_yi.clone().sum::<Value>();
            exp_yi.map(|y_ij| y_ij / &exp_sum).collect()
        })
        .collect()
}

pub fn sigmoid(y: &[Vec<Value>]) -> Vec<Vec<Value>> {
    y.iter()
        .map(|y_i| y_i.iter().map(|y_ij| y_ij.sigmoid()).collect())
        .collect()
}
