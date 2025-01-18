use ferrograd::engine::Value;

fn main() {
    let a = Value::new_with_name(5.6, 'a');
    let b = Value::new_with_name(10.8, 'b');

    let c = Value::new_with_name(-15.12, 'c');
    let d = Value::new_with_name(2.5, 'd');

    let e = (&a + &b) / 50.0;
    let e = e.with_name('e');

    let f = (&d - &c) * 5.5625;
    let f = f.with_name('f');

    let g = (e * f).leaky_relu().with_name('g');
    g.backward();

    println!("{}", g.tree());
}
