use super::tensor::Tensor1D;

pub trait Layer {
    fn parameters(&self) -> Vec<Tensor1D>;
    fn call(&self, x: Tensor1D) -> Tensor1D;
}

#[derive(Debug)]
pub struct Linear {
    weight: Tensor1D,
    bias: Tensor1D
}

impl Linear {
    pub fn new(size: (usize, usize)) -> Self {
        Self {
            weight: Tensor1D::zeros(size.0), // random
            bias: Tensor1D::ones(1)
        }
    }
}

impl Layer for Linear {
    fn parameters(&self) -> Vec<Tensor1D> {
        vec![self.weight, self.bias]
    }

    fn call(&self, x: Tensor1D) -> Tensor1D {
        x * self.weight + self.bias
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_new() { 
        let nn = Linear::new((2, 2));
        assert_eq!(nn.weight, vec![vec![0.0, 0.0], vec![0.0, 0.0]]);
        assert_eq!(nn.bias, vec![vec![0.0], vec![0.0]])
    }
}