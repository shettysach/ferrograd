mod engine;
use engine::Value;

fn main() {
    let a = Value::new(1.0);
    let b = Value::new(2.0);
    let c = Value::new(3.0);
    let d = Value::new(4.0);

    let e = &a * &b;
    let f = &c + &d;
    let g = &e * &f;

    g.backward();
    println!("{}", g.tree());
}
