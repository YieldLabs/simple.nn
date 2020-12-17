use super::tensor::Tensor1D;

#[derive(Debug)]
pub struct SGD {
    parameters: Vec<Tensor1D>,
    lr: Tensor1D
}

impl SGD {
    pub fn new(parameters: Vec<Tensor1D>, lr: f32) -> Self {
        Self {
            parameters: parameters,
            lr: Tensor1D::new(vec![lr]),
        }
    }

    pub fn zeros(&self) {
        for tensor in self.parameters {
            tensor.grad = None;
        }
    }

    pub fn step(&self) {
        for tensor in self.parameters {
            tensor = tensor - tensor * self.lr;
        }
    }
}