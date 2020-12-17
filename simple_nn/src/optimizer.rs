use super::tensor::Tensor1D;
use super::shared::Optimizer;

#[derive(Debug, Clone)]
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
}

impl Optimizer for SGD {
    fn zeros(self) {
        for mut tensor in self.parameters {
            tensor.grad = None;
        }
    }

    fn step(self) {
        for mut tensor in self.parameters {
            tensor = tensor.clone() - tensor.clone() * self.lr.clone();
        }
    }
}