use super::tensor::Tensor1D;

pub struct MSE {}

impl MSE {
    pub fn call(y_hat: Tensor1D, y: Tensor1D) -> Tensor1D {
        (y - y_hat).pow(2.0).mean()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse_loss() {
        let t1 = Tensor1D::new(vec![4.0, 3.0]);
        let t2 = Tensor1D::new(vec![1.0, 2.0]);
        
        let loss = MSE::call(t1, t2);
        assert_eq!(loss, Tensor1D::new(vec![5.0]));
    }
}