#[inline]
fn mse_loss(x: Tensor, y: Tensor) {
    x.dot(y)
}

#[inline]
fn bce_loss(x: Tensor, y: Tensor) {
    x.dot(y)
}