extern crate nalgebra as na;

use na::{DMatrix, DVector};
// use ndarray::prelude::*;
use rand::prelude::*;
use rand_distr::StandardNormal;
use std::collections::HashMap;

/*
Desired features:
1. allow training with different loss functions including custom ones
2. first class support for csv files, easy to use normal csvs (not wrapped) to train
3. allow transforms and inverse transform data including custom transforms
4. first class support to query internals such as learned embeddings, mapping
   values to internal integer indices
5. simple API for integrating in other programs
6. a plug and play command line interface
7. speed
*/

pub trait LossFn {}

pub trait Algo {
    fn fit(&mut self);
}

pub trait ErrorFn {
    fn call(&mut self, pred: f64, actual: f64) -> f64;
    fn grad(&mut self, pred: f64, actual: f64) -> f64;
    fn grad_user(&mut self, pred: f64, actual: f64, p_user: f64, q_item: f64) -> f64;
    fn grad_item(&mut self, pred: f64, actual: f64, p_user: f64, q_item: f64) -> f64;
}

pub struct MeanSquaredError {}

impl ErrorFn for MeanSquaredError {
    fn call(&mut self, pred: f64, actual: f64) -> f64 {
        (pred - actual) * (pred - actual)
    }

    fn grad(&mut self, pred: f64, actual: f64) -> f64 {
        2.0 * (pred - actual)
    }

    fn grad_user(&mut self, pred: f64, actual: f64, _p_user: f64, q_item: f64) -> f64 {
        2.0 * (pred - actual) * q_item
    }

    fn grad_item(&mut self, pred: f64, actual: f64, p_user: f64, _q_item: f64) -> f64 {
        2.0 * (pred - actual) * p_user
    }
}

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

pub struct Dataset {
    pub user_to_idx: HashMap<String, usize>,
    pub item_to_idx: HashMap<String, usize>,
    pub users: Vec<String>,
    pub items: Vec<String>,
    pub values: Vec<f64>,
}

impl Dataset {
    pub fn new<T>(data: T) -> Self
    where
        T: Iterator<Item = (String, String, f64)>,
    {
        let mut user_to_idx = HashMap::new();
        let mut item_to_idx = HashMap::new();
        let mut users = vec![];
        let mut items = vec![];
        let mut values = vec![];

        for (user, item, value) in data {
            if !user_to_idx.contains_key(&user) {
                user_to_idx.insert(user.clone(), user_to_idx.len());
            }

            if !item_to_idx.contains_key(&item) {
                item_to_idx.insert(item.clone(), item_to_idx.len());
            }

            users.push(user);
            items.push(item);
            values.push(value);
        }

        Self {
            user_to_idx,
            item_to_idx,
            users,
            items,
            values,
        }
    }
}

pub struct CsvReader {
    data: Vec<(String, String, f64)>,
}

impl CsvReader {
    pub fn new(path: &str, cols: (usize, usize, usize), delimiter: u8, has_headers: bool) -> Self {
        let (user_col, item_col, value_col) = cols;

        let mut rdr = csv::ReaderBuilder::new()
            .delimiter(delimiter)
            .has_headers(has_headers)
            .from_path(path)
            .unwrap();

        let mut data = vec![];

        for result in rdr.records() {
            let record = result.unwrap();
            let user = record.get(user_col).unwrap().to_owned();
            let item = record.get(item_col).unwrap().to_owned();
            let value: f64 = record.get(value_col).unwrap().parse().unwrap();
            data.push((user, item, value));
        }

        Self { data }
    }
}

impl IntoIterator for CsvReader {
    type Item = (String, String, f64);
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use crate::{CsvReader, Dataset, Svd};

    #[test]
    fn it_works() {
        let csv_reader = CsvReader::new("./test.csv", (0, 1, 2), b'\t', false);
        let dataset = Dataset::new(csv_reader.into_iter());
        let mut svd = Svd::default();
        svd.fit(dataset);
        println!("{:?}", svd.user_factors);
    }
}
