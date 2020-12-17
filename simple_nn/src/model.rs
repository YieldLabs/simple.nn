use super::nn::{Linear, Layer};
use super::tensor::Tensor1D;

trait Model {
    fn parameters(&self) -> Vec<Tensor1D>;
    fn forward(&self, x: Tensor1D) -> Tensor1D;
}

struct Dummy {
    x1: Linear,
    x2: Linear
}

impl Dummy {
    pub fn new(input_size: (usize, usize)) -> Self {
        Self {
            x1: Linear::new(input_size),
            x2: Linear::new((1, 10))
        }
    }
}

impl Model for Dummy {
    fn parameters(&self) -> Vec<Tensor1D> {
        let v = Vec::new();
        v.extend(self.x1.parameters());
        v.extend(self.x2.parameters());
        v
    }

    fn forward(&self, x: Tensor1D) -> Tensor1D {
        x = self.x1.call(x);
        x = x.relu();
        x = self.x2.call(x);
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_model() {
        let epochs = 10;

        let X = Tensor1D::new(vec![0.0, 1.0, 1.0]);
        let y = Tensor1D::new(vec![1.0, 1.0, 0.0]);
        
        let model = Dummy::new(X.shape);
        let outputs = model::forward(X);

        // let sgd = SGD::new(model.parameters(), 0.001);

        // for i in 0..epochs {
        //     let outputs = model::forward(X);

        //     sgd::zeros();

        //     let loss = MSE::call(outputs, y);
        
        //     loss::backward();
        //     sgd::step();
        // }
    }
}

