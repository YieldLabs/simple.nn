
struct MSE {}
impl MSE {
    fn new() {}

    fn call(y_hat: Tensor, y: Tensor) -> Tensor {
        y_hat.clone()
    }
}

struct BCE {}
impl BCE {
    fn new() {}
    fn call(y_hat: Tensor, y: Tensor) -> Tensor {
        y_hat.clone() 
    }
}