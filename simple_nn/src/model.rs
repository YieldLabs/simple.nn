struct Dummy {}

impl Dummy {
    fn new() {
        x1 = Linear()
        x2 = Linear()
    }

    fn forward(&self, x: Tensor) -> Tensor {
        x = x1.call(x)
        x = x2.call(x)
        x
    }
}

