
struct SGD {
    tensors: Vec<Tensor>
    lr: f64
}

impl SGD {
    fn new(tensors: Vec<Tensor>, lr: f64) -> Self {
        Self {
            tensors: tensors,
            lr: lr 
        }
    }

    fn step(&self) -> {
        for tensor in self.tensors {
            tensor.data -= self.lr * tensor.grad
        }
    }
}