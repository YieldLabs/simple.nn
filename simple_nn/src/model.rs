struct Dummy {}

impl Dummy {
    fn new(input_size: (usize, usize)) {
        self.x1 = Linear(input_size);
        self.x2 = Linear((1, 10));
    }

    fn forward(&self, x: Tensor) -> Tensor {
        x = self.x1.call(x);
        x = self.x2.call(x);
    }

    fn predict(x: Tensor) -> Tensor {
        x.clone();
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
        let criterion = MSELoss::new();

        let optimizer = SGD(model, lr=0.001, momentum=0.9);

        for i in 0..10 {
            let outputs = model::forward(data);

            optimizer::zeros();

            let loss = criterion(outputs, labels);
        
            loss::backward();
            optimizer::step();
        }

        assert_eq!(model::predict(vec![vec![1.0, 0.0, 2.5]]), 2.0);
    }
}

