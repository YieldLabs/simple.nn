use std::f64::consts::E;

#[inline]
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + E.powf(-x))
}

#[inline]
pub fn relu(x: f64) -> f64 {
    f64::max(0.0, x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu() {
        assert_eq!(relu(-1.0), 0.0);
        assert_eq!(relu(5.0), 5.0);
    }

    #[test]
    fn test_sigmoid() {
        assert_eq!(sigmoid(-1.0), 0.2689414213699951);
        assert_eq!(sigmoid(5.0), 0.9933071490757153);
    }
}