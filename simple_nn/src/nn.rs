use super::tensor::Tensor;

#[derive(Debug)]
pub struct Linear {
    weight: Tensor,
    bias: Tensor
}

impl Linear {
    pub fn new(size: (usize, usize)) -> Self {
        Self {
            weight: Tensor::zeros(size),
            bias: Tensor::ones(size)
        }
    }

    pub fn call(&self, x: Tensor) -> Tensor {
        return x.dot(self.weight).sum(self.bias);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_new() { 
        let nn = Linear::new((2, 2));
        assert_eq!(nn.weight, vec![vec![0.0, 0.0], vec![0.0, 0.0]]);
        assert_eq!(nn.bias, vec![vec![0.0, 0.0], vec![0.0, 0.0]])
    }
}