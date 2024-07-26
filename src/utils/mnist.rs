pub fn print_mnist_image(image: &[u8; 28 * 28]) {
    for row in 0..28 {
        for col in 0..28 {
            if image[row * 28 + col] == 0 {
                print!("□ ");
            } else {
                print!("■ ");
            }
        }
        println!();
    }
}
