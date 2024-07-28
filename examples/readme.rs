use ferrograd::engine::Value;

fn main() {
    let a = Value::new(-4.);
    let b = Value::new(2.);

    let mut c = &a + &b;
    let mut d = &a * &b + &b.pow(3.);

    c += &c + 1.;
    c += 1. + &c + (-&a);
    d += &d * 2. + (&b + &a).relu();
    d += 3. * &d + (&b - &a).relu();

    let e = &c - &d;
    let f = e.pow(2.);

    let mut g = &f / 2.;
    g += 10. / &f;

    println!("g.data = {:.4}", g.borrow().data);
    g.backward();

    println!("a.grad = {:.4}", a.borrow().grad);
    println!("b.grad = {:.4}", b.borrow().grad);
}
