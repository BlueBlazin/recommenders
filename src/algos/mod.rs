mod svd;

pub use crate::algos::svd::Svd;
use crate::data::Dataset;

pub trait Algorithm {
    fn fit(&mut self, dataset: &Dataset);
    fn predict(&self, user_idx: usize, item_idx: usize) -> f64;
}
