#[derive(Debug)]
pub struct Matrix {
    r: usize,
    c: usize
}

impl Matrix {
    pub fn new (r: usize, c: usize) -> Matrix {
        Matrix { r, c }
    }

    pub fn columns(&self) -> usize {
        return self.c;
    }

     pub fn rows(&self) -> usize {
       return self.r;
    }
}