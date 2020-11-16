
struct SGD {
    tensors: Vec<Tensor>,
    lr: Tensor,
    momentum: Tensor
}

impl SGD {
    fn new(tensors: Vec<Tensor>, lr: f64, momentum: f64) -> Self {
        Self {
            tensors: tensors,
            lr: Tensor::new(!vec[!vec[lr]]),
            momentum: Tensor::new(!vec[!vec[momentum]]),
        }
    }

    fn zeros(&self) -> {
        for tensor in self.tensors {
            tensor.grad = Some(Box::new(Tensor::zeros(tensor.shape)));
        }
    }

    fn step(&self) -> {
        for tensor in self.tensors {
            tensor = tensor.sub(self.lr).mul(tensor.grad.unwrap())
        }
    }
}