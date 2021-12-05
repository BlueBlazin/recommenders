use crate::data::Dataset;
use crate::similarities::Similarity;
use na::DMatrix;

pub struct Knn {
    pub k: usize,
    pub min_k: usize,
    pub user_based: bool,
    pub similarity: Similarity,
    sim: Option<DMatrix<f64>>,
}

impl Knn {
    pub fn new(k: usize, min_k: usize, user_based: bool, similarity: Similarity) -> Self {
        Self {
            sim: None,
            k,
            min_k,
            user_based,
            similarity,
        }
    }

    fn compute_similarities(&mut self, dataset: &Dataset) {}
}
