use crate::algos::Algorithm;
use crate::data::Dataset;
use crate::similarities::{cosine, msd, Similarity};
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

    fn compute_similarities(&mut self, dataset: &Dataset) -> DMatrix<f64> {
        let mut matrix = DMatrix::zeros(dataset.num_users, dataset.num_items);

        for i in 0..dataset.len() {
            let user = dataset.user_to_idx[&dataset.users[i]];
            let item = dataset.item_to_idx[&dataset.items[i]];
            let value = dataset.values[i];
            matrix[(user, item)] = value;
        }

        if !self.user_based {
            matrix.transpose_mut();
        }

        match self.similarity {
            Similarity::Cosine => cosine(matrix),
            Similarity::Msd => msd(matrix),
        }
    }
}

impl Algorithm for Knn {
    fn fit(&mut self, dataset: &Dataset) {
        self.sim = Some(self.compute_similarities(dataset));
    }

    fn predict(&self, user_idx: usize, item_idx: usize) -> f64 {
        todo!()
    }
}
