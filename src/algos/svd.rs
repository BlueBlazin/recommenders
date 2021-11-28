use crate::algos::Algorithm;
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
    pub verbose: bool,
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
        verbose: bool,
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
            verbose,
        }
    }

    pub fn fit_with(&mut self, dataset: &Dataset, mut error_fn: impl ErrorFn) {
        let num_users = dataset.user_to_idx.len();
        let num_items = dataset.item_to_idx.len();
        let num_examples = dataset.values.len();

        let randn = |_, _| thread_rng().sample(StandardNormal);

        let mut user_factors = DMatrix::from_fn(num_users, self.num_factors, randn);
        let mut item_factors = DMatrix::from_fn(num_items, self.num_factors, randn);

        let mut user_bias = DVector::zeros(num_users);
        let mut item_bias = DVector::zeros(num_items);

        // iterate over num epochs
        for epoch in 1..=self.num_epochs {
            let mut tot_epoch_err = 0.0;
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

                tot_epoch_err += error_fn.call(pred, actual);

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

            if self.verbose {
                println!(
                    "Average error after epoch {}: {}",
                    epoch,
                    tot_epoch_err / num_examples as f64,
                );
            }
        }

        self.user_bias = Some(user_bias);
        self.user_factors = Some(user_factors);
        self.item_bias = Some(item_bias);
        self.item_factors = Some(item_factors);
    }
}

impl Algorithm for Svd {
    fn fit(&mut self, dataset: &Dataset) {
        self.fit_with(dataset, MeanSquaredError {});
    }

    fn predict(&self, user_idx: usize, item_idx: usize) -> f64 {
        let user_vec = self.user_factors.as_ref().unwrap().row(user_idx);
        let item_vec = self.item_factors.as_ref().unwrap().row(item_idx);
        let bu = self.user_bias.as_ref().unwrap()[user_idx];
        let bi = self.item_bias.as_ref().unwrap()[item_idx];

        bu + bi + user_vec.dot(&item_vec)
    }
}

impl Default for Svd {
    fn default() -> Self {
        Svd::new(20, 100, 0.005, 0.005, 0.003, 0.003, true, false)
    }
}

#[cfg(test)]
mod tests {
    use crate::algos::{Algorithm, Evaluate, Svd};
    use crate::data::{CsvReader, Dataset};
    use crate::metrics::MetricType;

    #[test]
    fn test_fit() {
        let csv_reader = CsvReader::new("./test.csv", (0, 1, 2), b'\t', false);
        let dataset = Dataset::new(csv_reader.into_iter());
        let mut svd = Svd::default();
        svd.fit(&dataset);
    }

    #[test]
    fn test_predict() {
        let csv_reader = CsvReader::new("./test.csv", (0, 1, 2), b'\t', false);
        let dataset = Dataset::new(csv_reader.into_iter());
        let mut svd = Svd::default();
        svd.fit(&dataset);
        let _pred = svd.predict(
            dataset.user_to_idx[&dataset.users[0]],
            dataset.item_to_idx[&dataset.items[0]],
        );
    }

    #[test]
    fn test_evaluate() {
        let csv_reader = CsvReader::new("./test.csv", (0, 1, 2), b'\t', false);
        let dataset = Dataset::new(csv_reader.into_iter()).shuffle();
        let (trainset, testset) = dataset.train_test_split(80);

        let mut svd = Svd::default();
        svd.fit(&trainset);
        let metrics = svd.evaluate(
            &testset,
            vec![
                MetricType::Rmse,
                MetricType::Mae,
                MetricType::Mape,
                MetricType::Smape,
            ],
        );

        println!("{:?}", metrics);
    }
}
