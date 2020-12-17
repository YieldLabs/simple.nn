use super::tensor::Tensor1D;
use super::shared::Layer;

#[derive(Debug)]
pub struct Linear {
    weight: Tensor1D,
    bias: Tensor1D
}

impl Linear {
    pub fn new(size: (usize, usize)) -> Self {
        Self {
            weight: Tensor1D::uniform(size.0),
            bias: Tensor1D::ones(1)
        }
    }
}

impl Layer for Linear {
    fn parameters(self) -> Vec<Tensor1D> {
        vec![self.weight, self.bias]
    }

    fn call(self, x: Tensor1D) -> Tensor1D {
        x * self.weight + self.bias
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_new() { 
        let nn = Linear::new((2, 2));
        assert_eq!(nn.weight.shape, (1, 2));
        assert_eq!(nn.bias, Tensor1D::ones(1))
    }
}