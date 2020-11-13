
struct SGD {
    tensors: Vec<Tensor>
    lr: f64
}

impl SGD {
    pub fn new(tensors: Vec<Tensor>, lr: f64) -> Self {
        Self {
            tensors: tensors,
            lr: lr 
        }
    }

    pub fn step(&self) -> {
        for tensor in self.tensors {
            tensor.data -= self.lr * tensor.grad
        }
    }
}