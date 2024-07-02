mod engine;
use engine::Value;

fn main() {
    let a = Value::new(5.0);
    let a2 = Value::new(5.0);
    let b = &a.pow(2.);
    let c = &a2 * &a2;

    b.backward();
    println!("{}", b.tree());

    c.backward();
    println!("{}", c.tree());
}
