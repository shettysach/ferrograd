mod engine;
use engine::Value;

fn main() {
    let a = Value::new(3.0);
    let b = Value::new(5.0);
    let c = (4.0 / &a + &b).pow(0.0);
    let d = a.pow(3.0) + &c;

    println!("{:#?}", c);
    println!("{:#?}", d);
}
