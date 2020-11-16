#[derive(Debug)]
struct Linear {
    w: Tensor,
    b: Tensor
}

impl Linear {
    fn new(size: (usize, usize)) -> Self {
        Self {
            w: Tensor::zeros(size),
            b: Tensor::zeros(size)
        }
    }

    fn call(&self, x: Tensor) -> Tensor {
        return x.dot(self.w).sum(self.b);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_new() { 
        let nn = Linear::new((2, 2))
        assert_eq!(nn.w, [[0.0, 0.0], [0.0, 0.0]]);
        assert_eq!(nn.b, [[0.0, 0.0], [0.0, 0.0]])
    }
}