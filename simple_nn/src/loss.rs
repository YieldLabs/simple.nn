use super::tensor::Tensor;

pub struct MSE {}

impl MSE {
    pub fn new() {}

    fn call(y_hat: Tensor, y: Tensor) -> Tensor {
        y.sub(y_hat).pow().mean()
    }
}