
#[derive(Debug)]
struct Dense {
    w: Tensor,
    b: Tensor
}

impl Dense {
    pub fn new(size: (usize, usize)) -> Self {
        Self {
            w: Tensor::zeros(size),
            b: Tensor::zeros(size)
        }
    }

    pub fn call(&self, x: Tensor) -> {
        println!("{} {}", self.w, self.b);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_new() { 
        let nn = Dense::new((2, 2))
        assert_eq!(nn.w, [[0.0, 0.0], [0.0, 0.0]]);
        assert_eq!(nn.b, [[0.0, 0.0], [0.0, 0.0]])
    }
}