mod svd;

pub use crate::algos::svd::Svd;
use crate::data::Dataset;

pub trait Algorithm {
    fn fit(&mut self, dataset: Dataset);
}
