use micrograd::Value;

fn main() {
    let x1 = sample().relu();
    x1.backward();
    println!("{}", &x1.tree());

    let x2 = sample().tanh();
    x2.backward();
    println!("{}", &x2.tree());

    let x3 = sample().tanh();
    x3.backward();
    println!("{}", &x3.tree());
}

#[allow(dead_code)]
fn example() {
    let a = Value::new(-4.).with_name("a");
    let b = Value::new(2.).with_name("b");

    let mut c = (&a + &b).with_name("c");

    let mut d = (&a * &b + &b.pow(3.)).with_name("c");

    c += &c + 1.;
    c += 1. + &c + (-&a);
    d += &d * 2. + (&b + &a).relu();
    d += 3. * &d + (&b - &a).relu();

    let e = (&c - &d).with_name("e");

    let f = e.pow(2.).with_name("f");

    let mut g = (&f / 2.).with_name("g");
    g += 10. / &f;

    g.backward();
    println!("{}", g.tree());
}

#[allow(dead_code)]
fn sample() -> Value {
    let a = Value::new(-4.0).with_name("a");
    let b = Value::new(2.0).with_name("b");

    let c = (&a + &b + 15.).with_name("c");
    c
}
