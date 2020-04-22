use simple_nn::activation::{sigmoid};

#[test]
fn sigmoid_activation() {
    let data = sigmoid(0.334);
    assert_eq!(data, 0.5827323189732048);
}