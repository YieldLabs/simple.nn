use std::ops::{Add, Div, Mul, Sub};

#[derive(Debug, Clone)]
pub struct Tensor1D {
    pub data: Vec<f32>,
    pub grad: Option<Box<Tensor1D>>,
    pub shape: (usize, usize),
}

impl Tensor1D {
    pub fn new(data: Vec<f32>) -> Tensor1D {
        Tensor1D {
            data: data.clone(),
            grad: None,
            shape: (1, data.len())
        }
    }

    pub fn ones(size: usize) -> Tensor1D {
        let data = vec![1.0; size];

        Tensor1D {
            data: data.clone(),
            grad: None,
            shape: (1, data.len())
        }
    }

    pub fn zeros(size: usize) -> Tensor1D {
        let data = vec![0.0; size];
        
        Tensor1D {
            data: data.clone(),
            grad: None,
            shape: (1, data.len())
        }
    }

    pub fn backward(self) {
        match self.grad {
            None => self.grad = Some(Box::new(Tensor1D::ones(self.shape.1)))
        }
    }

    pub fn pow(self, x: Tensor1D) -> Tensor1D {
        Tensor1D {
            data: self.data.clone(),
            grad: None,
            shape: (1, self.data.len())
        }
    }

    pub fn mean(self, x: Tensor1D) -> Tensor1D {
        Tensor1D {
            data: self.data.clone(),
            grad: None,
            shape: (1, self.data.len())
        }
    }

    pub fn relu(self) -> Tensor1D {
        Tensor1D {
            data: self.data.clone(),
            grad: None,
            shape: (1, self.data.len())
        }
    }
}

impl Add for Tensor1D {
    type Output = Tensor1D;

    fn add(self, x: Tensor1D) -> Tensor1D {
        Tensor1D {
            data: self.data.clone(),
            grad: None,
            shape: (1, self.data.len())
        }
    }
}

impl Mul for Tensor1D {
    type Output = Tensor1D;
    
    fn mul(self, x: Tensor1D) -> Tensor1D {
        Tensor1D {
            data: self.data.clone(),
            grad: None,
            shape: (1, self.data.len())
        }
    }
}

impl Sub for Tensor1D {
    type Output = Tensor1D;
    
    fn sub(self, x: Tensor1D) -> Tensor1D {
        Tensor1D {
            data: self.data.clone(),
            grad: None,
            shape: (1, self.data.len())
        }
    }
}

impl Div for Tensor1D {
    type Output = Tensor1D;
    
    fn div(self, x: Tensor1D) -> Tensor1D {
        Tensor1D {
            data: self.data.clone(),
            grad: None,
            shape: (1, self.data.len())
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