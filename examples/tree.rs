use ferrograd::engine::Value;

fn main() {
    let a = Value::new(5.6).with_name("a");
    let b = Value::new(10.8).with_name("b");

    let c = Value::new(-15.12).with_name("c");
    let d = Value::new(2.5).with_name("d");

    let e = (&a + &b) / 50.0;
    let e = e.with_name("e");

    let f = (&d - &c) * 5.5625;
    let f = f.with_name("f");

    let g = (e * f).leaky_relu().with_name("g");
    g.backward();

    println!("{}", g.tree());
}
