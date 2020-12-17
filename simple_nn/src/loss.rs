use super::tensor::Tensor1D;

pub struct MSE {}

impl MSE {
    pub fn call(y_hat: Tensor1D, y: Tensor1D) -> Tensor1D {
        (y - y_hat).pow().mean()
    }
}