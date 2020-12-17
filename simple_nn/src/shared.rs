use super::tensor::Tensor1D;

pub trait Model {
    fn parameters(self) -> Vec<Tensor1D>;
    fn forward(self, x: Tensor1D) -> Tensor1D;
}

pub trait Layer {
    fn parameters(self) -> Vec<Tensor1D>;
    fn call(self, x: Tensor1D) -> Tensor1D;
}

pub trait Optimizer {
    fn zeros(self);
    fn step(self);
}