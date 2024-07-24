use super::engine::Value;
use std::fs::File;
use std::io::{BufRead, BufReader};

/**
Reads a CSV file __containing only numeric fields__ and returns a tuple (`xs`, `ys`).
- `filepath`: Path of the CSV file.
- `xnum`: Number of columns to read as features. (`xs`).
- `ynum`: Number of columns to read as targets. (`ys`).
- `skip_rows`: Number of initial rows to skip in the CSV file.
*/
pub fn read_csv(
    filepath: &str,
    xnum: usize,
    ynum: usize,
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

            assert!(fields.len() >= xnum + ynum, "Not enough fields in line");

            let x_vec = (0..xnum)
                .map(|i| Value::new(fields[i].parse::<f64>().unwrap()))
                .collect();

            let y_vec = (0..ynum)
                .map(|i| Value::new(fields[xnum + i].parse::<f64>().unwrap()))
                .collect();

            (x_vec, y_vec)
        })
        .collect();

    (xs, ys)
}
