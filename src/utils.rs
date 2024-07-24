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

    let mut xs = vec![Vec::new(); xnum];
    let mut ys = vec![Vec::new(); ynum];

    reader.lines().skip(skip_rows).for_each(|line| {
        let line = line.unwrap();
        let fields: Vec<&str> = line.split(',').collect();

        assert!(fields.len() >= xnum + ynum, "Not enough fields in line");

        (0..xnum).for_each(|i| {
            let x_val = fields[i].parse::<f64>().unwrap();
            xs[i].push(Value::new(x_val));
        });

        (0..ynum).for_each(|i| {
            let y_val = fields[xnum + i].parse::<f64>().unwrap();
            ys[i].push(Value::new(y_val));
        });
    });

    (xs, ys)
}
