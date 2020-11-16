use super::nn::Linear;
use super::tensor::Tensor;

struct Dummy {
    x1: Linear,
    x2: Linear
}

impl Dummy {
    fn new(input_size: (usize, usize)) -> Self {
        Self {
            x1: Linear::new(input_size),
            x2: Linear::new((1, 10))
        }
    }

    fn forward(&self, x: Tensor) -> Tensor {
        x = self.x1.call(x);
        self.x2.call(x)
    }

    fn predict(x: Tensor) -> Tensor {
        x.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_model() { 
        let labels = Tensor::new(vec![vec![1.0], vec![1.0], vec![1.0]]);
        let data = Tensor::new(vec![vec![0.0], vec![1.0], vec![0.0]]);
        
        let model = Dummy::new(data.shape);
        let criterion = MSE::new();

        let optimizer = SGD(model, 0.001, 0.9);

        for i in 0..10 {
            let outputs = model::forward(data);

            optimizer::zeros();

            let loss = criterion::call(outputs, labels);
        
            loss::backward();
            optimizer::step();
        }

        assert_eq!(model::predict(vec![vec![1.0, 0.0, 2.5]]), 2.0);
    }
}

