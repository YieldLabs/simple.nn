
struct MSE {}

impl MSE {
    fn new() {}

    fn call(y_hat: Tensor, y: Tensor) -> Tensor {
        y.sub(y_hat).pow().div(y.shape.1);
    }
}