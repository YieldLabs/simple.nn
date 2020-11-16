
struct SGD {
    tensors: Vec<Tensor>,
    lr: Tensor
}

impl SGD {
    fn new(tensors: Vec<Tensor>, lr: f64) -> Self {
        Self {
            tensors: tensors,
            lr: Tensor::new(!vec[!vec[lr]])
        }
    }

    fn step(&self) -> {
        for tensor in self.tensors {
            tensor = tensor.sub(self.lr).mul(tensor.grad.unwrap())
        }
    }
}

struct Adam {}

impl Adam {
    fn new() -> {}

    fn step(&self) -> {

    }
}