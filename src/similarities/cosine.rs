use crate::data::Dataset;
use na::{DMatrix, DVector};
use std::collections::HashMap;

pub fn cosine(data: HashMap<usize, Vec<(usize, f64)>>, size: usize) -> DMatrix<f64> {
    let mut sim = DMatrix::zeros(size, size);

    sim
}
