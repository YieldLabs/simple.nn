use std::ops::{Add, Div, Mul, Sub};
use rand::distributions::{Distribution, Uniform};
use std::f32::consts::E;


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
            shape: (data.len(), 0),
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
        // topologycal sort and reverse apply ops fn
    }

    pub fn transpose(self) -> Self {
        self.clone()
    }

    pub fn dot(self, x: Tensor1D) -> Self {
        let n = usize::max(self.shape.0, x.shape.0);
        let mut data = vec![0.0; 1];

        if self.shape.0 == x.shape.0 {
            for i in 0..n {
                for k in 0..1 {
                    data[k] = data[k] + self.data[i] * x.data[i]
                }
            }
        } else {
            panic!("Check shape size");
        }

        Self::new(data)
    }

    pub fn neg(self) -> Self {
        Self::new(vec![-1.0]) * self
    }

    pub fn abs(self) -> Self {
        let n = self.shape.0;
        let mut data = Vec::with_capacity(n);

        for i in 0..n {
            data.push(self.data[i].abs());
        }

        Self::new(data)
    }

    pub fn exp(self) -> Self {
        let n = self.shape.0;
        let mut data = Vec::with_capacity(n);

        for i in 0..n {
            data.push(self.data[i].exp());
        }

        Self::new(data)
    }

    pub fn sqrt(self) -> Self {
        let n = self.shape.0;
        let mut data = Vec::with_capacity(n);

        for i in 0..n {
            data.push(self.data[i].sqrt());
        }

        Self::new(data)
    }

    pub fn log(self) -> Self {
        let n = self.shape.0;
        let mut data = Vec::with_capacity(n);

        for i in 0..n {
            data.push(self.data[i].log(E));
        }

        Self::new(data)
    }

    pub fn log10(self) -> Self {
        let n = self.shape.0;
        let mut data = Vec::with_capacity(n);

        for i in 0..n {
            data.push(self.data[i].log10());
        }

        Self::new(data)
    }

    pub fn sin(self) -> Self {
        let n = self.shape.0;
        let mut data = Vec::with_capacity(n);

        for i in 0..n {
            data.push(self.data[i].sin());
        }

        Self::new(data)
    }

    pub fn cos(self) -> Self {
        let n = self.shape.0;
        let mut data = Vec::with_capacity(n);

        for i in 0..n {
            data.push(self.data[i].cos());
        }

        Self::new(data)
    }

    pub fn tan(self) -> Self {
        let n = self.shape.0;
        let mut data = Vec::with_capacity(n);

        for i in 0..n {
            data.push(self.data[i].tan());
        }

        Self::new(data)
    }

    pub fn pow(self, exp: f32) -> Self {
        let n = self.shape.0;
        let mut data = Vec::with_capacity(n);

        for i in 0..n {
            data.push(self.data[i].powf(exp));
        }

        Self::new(data)
    }

    pub fn max(self) -> Self {
        let mut res = 0.0;
        let n = self.shape.0;

        for i in 0..n {
            if res < self.data[i] {
                res = self.data[i];
            }
        }

        Self::new(vec![res])
    }

    pub fn min(self) -> Self {
        let mut res = 0.0;
        let n = self.shape.0;

        for i in 0..n {
            if res > self.data[i] {
                res = self.data[i];
            }
        }

        Self::new(vec![res])
    }

    pub fn sum(self) -> Self {
        let mut res = 0.0;
        let n = self.shape.0;

        for i in 0..n {
            res += self.data[i];
        }

        Self::new(vec![res])
    }

    pub fn mean(self) -> Self {
        let n = self.shape.0 as f32;

        self.sum() / Self::new(vec![n])
    }

    pub fn relu(self) -> Self {
        let n = self.shape.0;
        let mut data = Vec::with_capacity(n);

        for i in 0..n {
            data.push(f32::max(0.0, self.data[i]));
        }

        Self::new(data)
    }

    pub fn sigmoid(self) -> Self {
        let n = self.shape.0;
        let mut data = Vec::with_capacity(n);

        for i in 0..n {
            data.push(1.0 / (1.0 + E.powf(-self.data[i])));
        }

        Self::new(data)
    }
}

impl Clone for Tensor1D {
    fn clone(&self) -> Self {
        Self::new(self.data.clone())
    }
}

impl Add for Tensor1D {
    type Output = Self;

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
    type Output = Self;
    
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
    type Output = Self;
    
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
    type Output = Self;
    
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
        assert_eq!(t1.pow(2.0), Tensor1D::new(vec![9.0, 9.0, 25.0]));
    }

    #[test]
    fn test_exp() {
        let t1 = Tensor1D::new(vec![3.0, 3.0, 5.0]);
        assert_eq!(t1.exp(), Tensor1D::new(vec![20.085537, 20.085537, 148.41316]));
    }

    #[test]
    fn test_sin() {
        let t1 = Tensor1D::new(vec![3.0, 3.0, 5.0]);
        assert_eq!(t1.sin(), Tensor1D::new(vec![0.14112, 0.14112, -0.9589243]));
    }

    #[test]
    fn test_cos() {
        let t1 = Tensor1D::new(vec![3.0, 3.0, 5.0]);
        assert_eq!(t1.cos(), Tensor1D::new(vec![-0.9899925, -0.9899925, 0.2836622]));
    }

    #[test]
    fn test_tan() {
        let t1 = Tensor1D::new(vec![3.0, 3.0, 5.0]);
        assert_eq!(t1.tan(), Tensor1D::new(vec![-0.14254655, -0.14254655, -3.380515]));
    }

    #[test]
    fn test_log() {
        let t1 = Tensor1D::new(vec![3.0, 3.0, 5.0]);
        assert_eq!(t1.log(), Tensor1D::new(vec![1.0986124, 1.0986124, 1.6094381]));
    }

    #[test]
    fn test_log10() {
        let t1 = Tensor1D::new(vec![3.0, 3.0, 5.0]);
        assert_eq!(t1.log10(), Tensor1D::new(vec![0.47712126, 0.47712126, 0.69897]));
    }

    #[test]
    fn test_neg() {
        let t1 = Tensor1D::new(vec![3.0, -1.0, -5.0]);
        assert_eq!(t1.neg(), Tensor1D::new(vec![-3.0, 1.0, 5.0]));
    }

    #[test]
    fn test_abs() {
        let t1 = Tensor1D::new(vec![-3.0, -3.0, 5.0]);
        assert_eq!(t1.abs(), Tensor1D::new(vec![3.0, 3.0, 5.0]));
    }

    #[test]
    fn test_dot() {
        let t1 = Tensor1D::new(vec![3.0, 2.0, 3.0]);
        let t2 = Tensor1D::ones(3);
        assert_eq!(t1.dot(t2), Tensor1D::new(vec![8.0]));
    }

    #[test]
    fn test_transpose() {
        let t1 = Tensor1D::new(vec![3.0, 2.0, 3.0]);
        assert_eq!(t1.transpose(), Tensor1D::new(vec![3.0, 2.0, 3.0]));
    }

    #[test]
    fn test_max() {
        let t1 = Tensor1D::new(vec![9.0, 2.0, 3.0, 2.0, 0.0]);
        assert_eq!(t1.max(), Tensor1D::new(vec![9.0]));
    }

    #[test]
    fn test_min() {
        let t1 = Tensor1D::new(vec![9.0, -1.0, 3.0, 2.0, 0.0]);
        assert_eq!(t1.min(), Tensor1D::new(vec![-1.0]));
    }

    #[test]
    fn test_sum() {
        let t1 = Tensor1D::new(vec![3.0, 2.0, 3.0, 2.0, 0.0]);
        assert_eq!(t1.sum(), Tensor1D::new(vec![10.0]));
    }

    #[test]
    fn test_mean() {
        let t1 = Tensor1D::new(vec![3.0, 2.0, 3.0, 2.0, 0.0]);
        assert_eq!(t1.mean(), Tensor1D::new(vec![2.0]));
    }

    #[test]
    fn test_relu() {
        let t1 = Tensor1D::new(vec![-3.0, 2.0, -3.0, 2.0, 0.0]);
        assert_eq!(t1.relu(), Tensor1D::new(vec![0.0, 2.0, 0.0, 2.0, 0.0]));
    }

    #[test]
    fn test_sigmoid() {
        let t1 = Tensor1D::new(vec![-3.0, 2.0, -3.0, 2.0, 0.0]);
        assert_eq!(t1.sigmoid(), Tensor1D::new(vec![0.047425877, 0.880797, 0.047425877, 0.880797, 0.5]));
    }
}