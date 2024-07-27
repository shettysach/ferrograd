use crate::engine::Value;
use std::fs::File;
use std::io::{BufRead, BufReader};

/**
Reads a CSV file __containing only numeric fields__ and returns a tuple (`xs`, `ys`).
- `filepath`: Path of the CSV file.
- `x_cols`: Columns to read as features. (`xs`).
- `y_cols`: Columns to read as targets. (`ys`).
- `skip_rows`: Number of initial rows to skip in the CSV file.
*/
pub fn read_csv(
    filepath: &str,
    x_cols: &[usize],
    y_cols: &[usize],
    skip_rows: usize,
) -> (Vec<Vec<Value>>, Vec<Vec<Value>>) {
    let file = File::open(filepath).unwrap();
    let reader = BufReader::new(file);

    let (xs, ys) = reader
        .lines()
        .skip(skip_rows)
        .map(|line| {
            let line = line.unwrap();
            let fields: Vec<&str> = line.split(',').collect();

            assert!(
                fields.len() >= x_cols.len() + y_cols.len(),
                "Not enough fields in the line"
            );

            let x_vec = x_cols
                .iter()
                .map(|i| Value::new(fields[*i].parse::<f64>().unwrap()))
                .collect();

            let y_vec = y_cols
                .iter()
                .map(|i| Value::new(fields[*i].parse::<f64>().unwrap()))
                .collect();

            (x_vec, y_vec)
        })
        .collect();

    (xs, ys)
}
