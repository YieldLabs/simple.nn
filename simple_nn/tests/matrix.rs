use simple_nn::matrix::Matrix;

#[test]
fn sigmoid_activation() {
    let matrix = Matrix::new(5, 3);
    assert_eq!(matrix.columns(), 3);
    assert_eq!(matrix.rows(), 5);
}