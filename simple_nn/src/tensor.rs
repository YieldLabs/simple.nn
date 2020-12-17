use std::ops::{Add, Div, Mul, Sub};
use rand::distributions::{Distribution, Uniform};

#[derive(Debug, PartialEq)]
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
            shape: (data.len(), 0)
        }
    }

    pub fn ones(size: usize) -> Self {
        Self::new(vec![1.0; size])
    }

    pub fn zeros(size: usize) -> Self {
        Self::new(vec![0.0; size])
    }

    pub fn uniform(size: usize) -> Self {
        let uniform = Uniform::from(-1.0..1.0);
        let mut rng = rand::thread_rng();
        
        let mut data = Vec::with_capacity(size);

        for _ in 0..size {
            data.push(uniform.sample(&mut rng));
        }

        Self::new(data)
    }

    pub fn backward(mut self) {
        self.grad = Some(Box::new(Tensor1D::ones(self.shape.0)));
    }

    pub fn pow(self) -> Self {
        self.clone() * self.clone()
    }

    pub fn mean(self) -> Self {
        let mut res = 0.0;
        
        for i in 0..self.shape.0 {
            res += self.data[i];
        }

        Self::new(vec![res / self.shape.0 as f32])
    }

    pub fn relu(self) -> Self {
        Self::new(self.data)
    }
}

impl Clone for Tensor1D {
    fn clone(&self) -> Self {
        Tensor1D::new(self.data.clone())
    }
}

impl Add for Tensor1D {
    type Output = Tensor1D;

    fn add(self, x: Tensor1D) -> Self {
        let n = usize::max(self.shape.0, x.shape.0);
        let mut data = vec![0.0; n];

        if self.shape.0 == 1 {
            for i in 0..n {
                for j in 0..1 {
                    data[i] = self.data[j] + x.data[i];
                }
            }
        }

        if x.shape.0 == 1 {
            for i in 0..n {
                for j in 0..1 {
                    data[i] = self.data[i] + x.data[j];
                }
            }
        }

        if self.shape.0 == x.shape.0 {
            for i in 0..n {
                data[i] = self.data[i] + x.data[i];
            }
        }
            
        Self::new(data)
    }
}

impl Mul for Tensor1D {
    type Output = Tensor1D;
    
    fn mul(self, x: Tensor1D) -> Self {
        let n = usize::max(self.shape.0, x.shape.0);
        let mut data = vec![0.0; n];

        if self.shape.0 == 1 {
            for i in 0..n {
                for j in 0..1 {
                    data[i] = self.data[j] * x.data[i];
                }
            }
        }

        if x.shape.0 == 1 {
            for i in 0..n {
                for j in 0..1 {
                    data[i] = self.data[i] * x.data[j];
                }
            }
        }

        if self.shape.0 == x.shape.0 {
            for i in 0..n {
                data[i] = self.data[i] * x.data[i];
            }
        }

        Self::new(data)
    }
}

impl Sub for Tensor1D {
    type Output = Tensor1D;
    
    fn sub(self, x: Tensor1D) -> Self {
        let n = usize::max(self.shape.0, x.shape.0);
        let mut data = vec![0.0; n];

        if self.shape.0 == 1 {
            for i in 0..n {
                for j in 0..1 {
                    data[i] = self.data[j] - x.data[i];
                }
            }
        }

        if x.shape.0 == 1 {
            for i in 0..n {
                for j in 0..1 {
                    data[i] = self.data[i] - x.data[j];
                }
            }
        }

        if self.shape.0 == x.shape.0 {
            for i in 0..n {
                data[i] = self.data[i] - x.data[i];
            }
        }
            
        Self::new(data)
    }
}


impl Div for Tensor1D {
    type Output = Tensor1D;
    
    fn div(self, x: Tensor1D) -> Self {
        let n = usize::max(self.shape.0, x.shape.0);
        let mut data = vec![0.0; n];

        if self.shape.0 == 1 {
            for i in 0..n {
                for j in 0..1 {
                    data[i] = self.data[j] / x.data[i];
                }
            }
        }

        if x.shape.0 == 1 {
            for i in 0..n {
                for j in 0..1 {
                    data[i] = self.data[i] / x.data[j];
                }
            }
        }

        if self.shape.0 == x.shape.0 {
            for i in 0..n {
                data[i] = self.data[i] / x.data[i];
            }
        }
            
        Self::new(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_new() {
        let data = vec![1.0, 2.0];
        let tensor = Tensor1D::new(data);

        assert_eq!(tensor.shape, (2, 0));
        assert_eq!(tensor.data, [1.0, 2.0]);
    }

    #[test]
    fn test_create_ones() {
        let tensor = Tensor1D::ones(3);

        assert_eq!(tensor.shape, (3, 0));
        assert_eq!(tensor.data, [1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_create_zeros() {
        let tensor = Tensor1D::zeros(3);

        assert_eq!(tensor.shape, (3, 0));
        assert_eq!(tensor.data, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_add() {
        let t1 = Tensor1D::ones(3);
        let t2 = Tensor1D::ones(1);
        assert_eq!(t1 + t2, Tensor1D::new(vec![2.0, 2.0, 2.0]));

        let t3 = Tensor1D::new(vec![2.0]);
        let t4 = Tensor1D::new(vec![3.0, 3.0, 5.0]);

        assert_eq!(t3 + t4, Tensor1D::new(vec![5.0, 5.0, 7.0]));
    }

    #[test]
    fn test_sub() {
        let t1 = Tensor1D::new(vec![4.0, 4.0, 4.0]);
        let t2 = Tensor1D::ones(1);
        assert_eq!(t1 - t2, Tensor1D::new(vec![3.0, 3.0, 3.0]));

        let t3 = Tensor1D::new(vec![2.0]);
        let t4 = Tensor1D::new(vec![3.0, 3.0, 5.0]);

        assert_eq!(t3 - t4, Tensor1D::new(vec![-1.0, -1.0, -3.0]));
    }

    #[test]
    fn test_div() {
        let t1 = Tensor1D::ones(3);
        let t2 = Tensor1D::new(vec![2.0]);
        assert_eq!(t1 / t2, Tensor1D::new(vec![0.5, 0.5, 0.5]));

        let t3 = Tensor1D::new(vec![2.0]);
        let t4 = Tensor1D::new(vec![3.0, 3.0, 5.0]);

        assert_eq!(t3 / t4, Tensor1D::new(vec![0.6666667, 0.6666667, 0.4]));
    }

    #[test]
    fn test_mul() {
        let t1 = Tensor1D::new(vec![2.0, 2.0, 3.0]);
        let t2 = Tensor1D::new(vec![2.0, 2.0, 3.0]);
        assert_eq!(t1 * t2, Tensor1D::new(vec![4.0, 4.0, 9.0]));

        let t3 = Tensor1D::new(vec![2.0]);
        let t4 = Tensor1D::new(vec![3.0, 3.0, 5.0]);

        assert_eq!(t3 * t4, Tensor1D::new(vec![6.0, 6.0, 10.0]));
    }

    #[test]
    fn test_pow() {
        let t1 = Tensor1D::new(vec![3.0, 3.0, 5.0]);
        assert_eq!(t1.pow(), Tensor1D::new(vec![9.0, 9.0, 25.0]));
    }
}