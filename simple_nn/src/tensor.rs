
#[derive(Debug, Clone)]
pub struct Tensor1D {
    pub data: Vec<f32>,
    pub grad: Option<Box<Tensor1D>>,
    pub shape: (usize, usize),
}

impl Tensor1D {
    pub fn new(data: Vec<f32>) -> Self {
        Self {
            data: data.clone(),
            grad: None,
            shape: (1, data.len())
        }
    }

    pub fn ones(size: usize) -> Self {
        let data = vec![1.0; size];

        Self {
            data: data.clone(),
            grad: None,
            shape: (1, data.len())
        }
    }

    pub fn zeros(size: usize) -> Self {
        let data = vec![0.0; size];
        
        Self {
            data: data.clone(),
            grad: None,
            shape: (1, data.len())
        }
    }

    pub fn backward(&self) {
        match self.grad {
            None => self.grad = Some(Box::new(Tensor1D::ones(self.shape.1)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_new() {
        let data = vec![1.0, 2.0];
        let tensor = Tensor1D::new(data);

        assert_eq!(tensor.shape, (1, 2));
        assert_eq!(tensor.data, [1.0, 2.0]);
    }

    #[test]
    fn test_create_ones() {
        let tensor = Tensor1D::ones(3);

        assert_eq!(tensor.shape, (1, 3));
        assert_eq!(tensor.data, [1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_create_zeros() {
        let tensor = Tensor1D::zeros(3);

        assert_eq!(tensor.shape, (1, 3));
        assert_eq!(tensor.data, [0.0, 0.0, 0.0]);
    }
}