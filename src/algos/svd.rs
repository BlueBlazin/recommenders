use crate::data::Dataset;
use crate::objectives::{ErrorFn, MeanSquaredError};
use na::{DMatrix, DVector};
use rand::prelude::*;
use rand_distr::StandardNormal;

pub struct Svd {
    pub user_bias: Option<DVector<f64>>,
    pub user_factors: Option<DMatrix<f64>>,
    pub item_bias: Option<DVector<f64>>,
    pub item_factors: Option<DMatrix<f64>>,
    pub num_factors: usize,
    pub num_epochs: usize,
    pub lr_user: f64,
    pub lr_item: f64,
    pub reg_user: f64,
    pub reg_item: f64,
    pub biased: bool,
}

impl Svd {
    pub fn new(
        num_factors: usize,
        num_epochs: usize,
        lr_user: f64,
        lr_item: f64,
        reg_user: f64,
        reg_item: f64,
        biased: bool,
    ) -> Self {
        Self {
            user_bias: None,
            user_factors: None,
            item_bias: None,
            item_factors: None,
            num_factors,
            num_epochs,
            lr_user,
            lr_item,
            reg_user,
            reg_item,
            biased,
        }
    }

    pub fn fit(&mut self, dataset: Dataset) {
        self.fit_with(dataset, MeanSquaredError {})
    }

    pub fn fit_with(&mut self, dataset: Dataset, mut error_fn: impl ErrorFn) {
        let num_users = dataset.user_to_idx.len();
        let num_items = dataset.item_to_idx.len();
        let num_examples = dataset.values.len();

        let randn = |_, _| thread_rng().sample(StandardNormal);

        let mut user_factors = DMatrix::from_fn(num_users, self.num_factors, randn);
        let mut item_factors = DMatrix::from_fn(num_items, self.num_factors, randn);

        let mut user_bias = DVector::zeros(num_users);
        let mut item_bias = DVector::zeros(num_items);

        // iterate over num epochs
        for epoch in 0..self.num_epochs {
            // iterate over all training examples
            for i in 0..num_examples {
                let user_idx = dataset.user_to_idx[&dataset.users[i]];
                let item_idx = dataset.item_to_idx[&dataset.items[i]];

                let user_vec = user_factors.row(user_idx);
                let item_vec = item_factors.row(item_idx);
                let bu = user_bias[user_idx];
                let bi = item_bias[item_idx];

                let pred = bu + bi + user_vec.dot(&item_vec);
                let actual = dataset.values[i];

                let error = error_fn.call(pred, actual);

                let (lr_u, lr_i) = (self.lr_user, self.lr_item);
                let (reg_u, reg_i) = (self.reg_user, self.reg_item);
                let grad_bias = error_fn.grad(pred, actual);

                if self.biased {
                    let (bu, bi) = (user_bias[user_idx], item_bias[item_idx]);
                    user_bias[user_idx] -= lr_u * (grad_bias + reg_u * bu);
                    item_bias[user_idx] -= lr_i * (grad_bias + reg_i * bi);
                }

                for f in 0..self.num_factors {
                    let p_user = user_factors[(user_idx, f)];
                    let q_item = item_factors[(item_idx, f)];

                    let grad_user = error_fn.grad_user(pred, actual, p_user, q_item);
                    let grad_item = error_fn.grad_item(pred, actual, p_user, q_item);

                    user_factors[(user_idx, f)] -= lr_u * (grad_user + reg_u * p_user);
                    item_factors[(item_idx, f)] -= lr_i * (grad_item + reg_i * q_item);
                }
            }
        }

        self.user_bias = Some(user_bias);
        self.user_factors = Some(user_factors);
        self.item_bias = Some(item_bias);
        self.item_factors = Some(item_factors);
    }
}

impl Default for Svd {
    fn default() -> Self {
        Svd::new(20, 100, 0.005, 0.005, 0.003, 0.003, true)
    }
}
