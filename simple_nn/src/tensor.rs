use std::ops::{Add, Div, Mul, Sub};
use rand::distributions::{Distribution, Uniform};
use std::f32::consts::E;

#[derive(Debug)]
pub enum Ops {
    T,
    DOT,
    ABS,
    EXP,
    SQRT,
    LOG,
    LOG10,
    SIN,
    COS,
    TAN,
    POW(f32),
    SUM,
    MEAN,
    RELU,
    SIGMOID,
    ADD,
    SUB,
    MUL,
    DIV,
    NEG,
    MIN,
    MAX
}

#[derive(Debug)]
pub struct Tensor1D {
    pub data: Vec<f32>,
    pub shape: (usize, usize),
    pub grad: Option<Box<Tensor1D>>,
    depends_on: (Option<Box<Tensor1D>>, Option<Box<Tensor1D>>),
    ops: Option<Box<Ops>>
}

impl Tensor1D {
    pub fn new(data: Vec<f32>) -> Tensor1D {
        Tensor1D {
            data: data.clone(),
            shape: (data.len(), 0),
            grad: None,
            depends_on: (None, None),
            ops: None
        }
    }

    pub fn new_with_deps(data: Vec<f32>, lhs: Option<Box<Tensor1D>>, rhs: Option<Box<Tensor1D>>, ops: Option<Box<Ops>>) -> Tensor1D {
        Tensor1D {
            data: data.clone(),
            shape: (data.len(), 0),
            grad: None,
            depends_on: (lhs, rhs),
            ops: ops
        }
    }

    pub fn ones(size: usize) -> Tensor1D {
        Tensor1D::new(vec![1.0; size])
    }

    pub fn zeros(size: usize) -> Tensor1D {
        Tensor1D::new(vec![0.0; size])
    }

    pub fn uniform(size: usize) -> Tensor1D {
        let uniform = Uniform::from(-1.0..1.0);
        let mut rng = rand::thread_rng();
        
        let mut data = Vec::with_capacity(size);

        for _ in 0..size {
            data.push(uniform.sample(&mut rng));
        }

        Tensor1D::new(data)
    }

    pub fn backward(mut self) {
        self.grad = Some(Box::new(Tensor1D::ones(self.shape.0)));
        // topologycal sort and reverse apply ops fn

    }

    pub fn transpose(self) -> Tensor1D {
        let t = self.clone();
        Tensor1D::new_with_deps(t.data, Some(Box::new(self)), None, Some(Box::new(Ops::T)))
    }

    pub fn dot(self, x: Tensor1D) -> Tensor1D {
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

        Tensor1D::new_with_deps(data, Some(Box::new(self)),  Some(Box::new(x)), Some(Box::new(Ops::DOT)))
    }

    pub fn abs(self) -> Tensor1D {
        let n = self.shape.0;
        let mut data = Vec::with_capacity(n);

        for i in 0..n {
            data.push(self.data[i].abs());
        }

        Tensor1D::new_with_deps(data, Some(Box::new(self)), None, Some(Box::new(Ops::ABS)))
    }

    pub fn exp(self) -> Tensor1D {
        let n = self.shape.0;
        let mut data = Vec::with_capacity(n);

        for i in 0..n {
            data.push(self.data[i].exp());
        }

        Tensor1D::new_with_deps(data, Some(Box::new(self)), None, Some(Box::new(Ops::EXP)))
    }

    pub fn sqrt(self) -> Tensor1D {
        let n = self.shape.0;
        let mut data = Vec::with_capacity(n);

        for i in 0..n {
            data.push(self.data[i].sqrt());
        }

        Tensor1D::new_with_deps(data, Some(Box::new(self)), None, Some(Box::new(Ops::SQRT)))
    }

    pub fn log(self) -> Tensor1D {
        let n = self.shape.0;
        let mut data = Vec::with_capacity(n);

        for i in 0..n {
            data.push(self.data[i].log(E));
        }

        Tensor1D::new_with_deps(data, Some(Box::new(self)), None, Some(Box::new(Ops::LOG)))
    }

    pub fn log10(self) -> Tensor1D {
        let n = self.shape.0;
        let mut data = Vec::with_capacity(n);

        for i in 0..n {
            data.push(self.data[i].log10());
        }

        Tensor1D::new_with_deps(data, Some(Box::new(self)), None, Some(Box::new(Ops::LOG10)))
    }

    pub fn sin(self) -> Tensor1D {
        let n = self.shape.0;
        let mut data = Vec::with_capacity(n);

        for i in 0..n {
            data.push(self.data[i].sin());
        }

        Tensor1D::new_with_deps(data, Some(Box::new(self)), None, Some(Box::new(Ops::SIN)))
    }

    pub fn cos(self) -> Tensor1D {
        let n = self.shape.0;
        let mut data = Vec::with_capacity(n);

        for i in 0..n {
            data.push(self.data[i].cos());
        }

        Tensor1D::new_with_deps(data, Some(Box::new(self)), None, Some(Box::new(Ops::COS)))
    }

    pub fn tan(self) -> Tensor1D {
        let n = self.shape.0;
        let mut data = Vec::with_capacity(n);

        for i in 0..n {
            data.push(self.data[i].tan());
        }

        Tensor1D::new_with_deps(data, Some(Box::new(self)), None, Some(Box::new(Ops::TAN)))
    }

    pub fn pow(self, exp: f32) -> Tensor1D {
        let n = self.shape.0;
        let mut data = Vec::with_capacity(n);

        for i in 0..n {
            data.push(self.data[i].powf(exp));
        }

        Tensor1D::new_with_deps(data, Some(Box::new(self)), None, Some(Box::new(Ops::POW(2.0))))
    }

    pub fn max(self) -> Tensor1D {
        let mut res = 0.0;
        let n = self.shape.0;

        for i in 0..n {
            if res < self.data[i] {
                res = self.data[i];
            }
        }

        Tensor1D::new_with_deps(vec![res], Some(Box::new(self)), None, Some(Box::new(Ops::MAX)))
    }

    pub fn min(self) -> Tensor1D {
        let mut res = 0.0;
        let n = self.shape.0;

        for i in 0..n {
            if res > self.data[i] {
                res = self.data[i];
            }
        }

        Tensor1D::new_with_deps(vec![res], Some(Box::new(self)), None, Some(Box::new(Ops::MIN)))
    }

    pub fn neg(self) -> Tensor1D {
        let n = self.shape.0;
        let mut data = Vec::with_capacity(n);

        for i in 0..n {
            data.push(-1.0 * self.data[i]);
        }

        Tensor1D::new_with_deps(data, Some(Box::new(self)), None, Some(Box::new(Ops::NEG)))
    }

    pub fn sum(self) -> Tensor1D {
        let mut res = 0.0;
        let n = self.shape.0;

        for i in 0..n {
            res += self.data[i];
        }

        Tensor1D::new_with_deps(vec![res], Some(Box::new(self)), None, Some(Box::new(Ops::SUM)))
    }

    pub fn mean(self) -> Tensor1D {
        let mut res = 0.0;
        let n = self.shape.0;

        for i in 0..n {
            res += self.data[i];
        }

        Tensor1D::new_with_deps(vec![res / n as f32], Some(Box::new(self)), None, Some(Box::new(Ops::MEAN)))
    }

    pub fn relu(self) -> Tensor1D {
        let n = self.shape.0;
        let mut data = Vec::with_capacity(n);

        for i in 0..n {
            data.push(f32::max(0.0, self.data[i]));
        }

        Tensor1D::new_with_deps(data, Some(Box::new(self)), None, Some(Box::new(Ops::RELU)))
    }

    pub fn sigmoid(self) -> Tensor1D {
        let n = self.shape.0;
        let mut data = Vec::with_capacity(n);

        for i in 0..n {
            data.push(1.0 / (1.0 + E.powf(-self.data[i])));
        }

        Tensor1D::new_with_deps(data, Some(Box::new(self)), None, Some(Box::new(Ops::SIGMOID)))
    }
}

impl Clone for Tensor1D {
    fn clone(&self) -> Tensor1D {
        Tensor1D::new(self.data.clone())
    }
}

impl PartialEq for Tensor1D {
    fn eq(&self, other: &Tensor1D) -> bool {
        self.data == other.data
    }
}

impl Add for Tensor1D {
    type Output = Tensor1D;

    fn add(self, x: Tensor1D) -> Tensor1D {
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
            
        Tensor1D::new_with_deps(data, Some(Box::new(self)), Some(Box::new(x)), Some(Box::new(Ops::ADD)))
    }
}

impl Mul for Tensor1D {
    type Output = Tensor1D;
    
    fn mul(self, x: Tensor1D) -> Tensor1D {
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

        Tensor1D::new_with_deps(data, Some(Box::new(self)), Some(Box::new(x)), Some(Box::new(Ops::MUL)))
    }
}

impl Sub for Tensor1D {
    type Output = Tensor1D;
    
    fn sub(self, x: Tensor1D) -> Tensor1D {
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
            
        Tensor1D::new_with_deps(data, Some(Box::new(self)), Some(Box::new(x)), Some(Box::new(Ops::SUB)))
    }
}


impl Div for Tensor1D {
    type Output = Tensor1D;
    
    fn div(self, x: Tensor1D) -> Tensor1D {
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
            
        Tensor1D::new_with_deps(data, Some(Box::new(self)), Some(Box::new(x)), Some(Box::new(Ops::DIV)))
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