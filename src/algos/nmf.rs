use crate::algos::Algorithm;
use crate::data::Dataset;
use na::{DMatrix, DVector};
use rand::prelude::*;
use rand_distr::Uniform;

pub struct Nmf {
    pub num_factors: usize,
    pub num_epochs: usize,
    pub biased: bool,
    pub lr_user: f64,
    pub lr_item: f64,
    pub reg_user: f64,
    pub reg_item: f64,
    init_low: f64,
    init_high: f64,
    verbose: bool,
    pub user_bias: Option<DVector<f64>>,
    pub user_factors: Option<DMatrix<f64>>,
    pub item_bias: Option<DVector<f64>>,
    pub item_factors: Option<DMatrix<f64>>,
}

impl Nmf {
    pub fn new(
        num_factors: usize,
        num_epochs: usize,
        biased: bool,
        lr_user: f64,
        lr_item: f64,
        reg_user: f64,
        reg_item: f64,
        init_low: f64,
        init_high: f64,
        verbose: bool,
    ) -> Self {
        Self {
            num_factors,
            num_epochs,
            biased,
            lr_user,
            lr_item,
            reg_user,
            reg_item,
            init_low,
            init_high,
            verbose,
            user_bias: None,
            user_factors: None,
            item_bias: None,
            item_factors: None,
        }
    }

    fn get_counts(&self, dataset: &Dataset) -> (Vec<usize>, Vec<usize>) {
        let mut user_to_count = vec![0; dataset.user_to_idx.len()];
        let mut item_to_count = vec![0; dataset.item_to_idx.len()];

        for i in 0..dataset.len() {
            let user_idx = dataset.user_to_idx[&dataset.users[i]];
            let item_idx = dataset.item_to_idx[&dataset.items[i]];
            user_to_count[user_idx] += 1;
            item_to_count[item_idx] += 1;
        }

        (user_to_count, item_to_count)
    }
}

impl Default for Nmf {
    fn default() -> Self {
        Self::new(15, 50, false, 0.005, 0.005, 0.06, 0.06, 1e-12, 1.0, false)
    }
}

impl Algorithm for Nmf {
    fn fit(&mut self, dataset: &Dataset) {
        let num_users = dataset.user_to_idx.len();
        let num_items = dataset.item_to_idx.len();
        let num_examples = dataset.values.len();

        let rand = |_, _| thread_rng().sample(Uniform::new(self.init_low, self.init_high));

        let mut user_factors = DMatrix::from_fn(num_users, self.num_factors, rand);
        let mut item_factors = DMatrix::from_fn(num_items, self.num_factors, rand);

        let mut user_bias = DVector::zeros(num_users);
        let mut item_bias = DVector::zeros(num_items);

        let mut user_num: DMatrix<f64>;
        let mut user_denom: DMatrix<f64>;
        let mut item_num: DMatrix<f64>;
        let mut item_denom: DMatrix<f64>;

        let (lr_u, lr_i) = (self.lr_user, self.lr_item);
        let (reg_u, reg_i) = (self.reg_user, self.reg_item);

        let (user_to_count, item_to_count) = self.get_counts(dataset);

        for epoch in 1..=self.num_epochs {
            user_num = DMatrix::zeros(num_users, self.num_factors);
            user_denom = DMatrix::zeros(num_users, self.num_factors);
            item_num = DMatrix::zeros(num_items, self.num_factors);
            item_denom = DMatrix::zeros(num_items, self.num_factors);

            let mut tot_epoch_err = 0.0;

            for i in 0..num_examples {
                let user_idx = dataset.user_to_idx[&dataset.users[i]];
                let item_idx = dataset.item_to_idx[&dataset.items[i]];

                // get factor vectors and biases
                let user_vec = user_factors.row(user_idx);
                let item_vec = item_factors.row(item_idx);
                let bu = user_bias[user_idx];
                let bi = item_bias[item_idx];

                // compute prediction
                let pred = bu + bi + user_vec.dot(&item_vec);
                let actual = dataset.values[i];

                // calculate error and add  to total
                let error = pred - actual;
                tot_epoch_err += error;

                // if biased perform SGD on biases
                if self.biased {
                    let (bu, bi) = (user_bias[user_idx], item_bias[item_idx]);
                    user_bias[user_idx] -= lr_u * (error + reg_u * bu);
                    item_bias[user_idx] -= lr_i * (error + reg_i * bi);
                }

                // compute numerators and denominators
                for f in 0..self.num_factors {
                    user_num[(user_idx, f)] += item_vec[f] * actual;
                    user_denom[(user_idx, f)] += item_vec[f] * pred;

                    item_num[(item_idx, f)] += user_vec[f] * actual;
                    item_num[(item_idx, f)] += user_vec[f] * pred;
                }
            }

            // update user factors following NMF update rule
            for u in 0..num_users {
                let count = user_to_count[u] as f64;
                for f in 0..self.num_factors {
                    // add regularization to denominator
                    user_denom[(u, f)] += reg_u * count * user_factors[(u, f)];
                    user_factors[(u, f)] += user_num[(u, f)] / user_denom[(u, f)];
                }
            }

            // update item factors following NMF update rule
            for i in 0..num_items {
                let count = item_to_count[i] as f64;
                for f in 0..self.num_factors {
                    // add regularization to denominator
                    item_denom[(i, f)] += reg_i * count * item_factors[(i, f)];
                    item_factors[(i, f)] += item_num[(i, f)] / item_denom[(i, f)];
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

        // set biases and factors on instance
        self.user_bias = Some(user_bias);
        self.user_factors = Some(user_factors);
        self.item_bias = Some(item_bias);
        self.item_factors = Some(item_factors);
    }

    fn predict(&self, user_idx: usize, item_idx: usize) -> f64 {
        let user_vec = self.user_factors.as_ref().unwrap().row(user_idx);
        let item_vec = self.item_factors.as_ref().unwrap().row(item_idx);
        let bu = self.user_bias.as_ref().unwrap()[user_idx];
        let bi = self.item_bias.as_ref().unwrap()[item_idx];

        bu + bi + user_vec.dot(&item_vec)
    }
}

#[cfg(test)]
mod tests {
    use crate::algos::{Algorithm, Evaluate, Nmf};
    use crate::data::{CsvReader, Dataset};
    use crate::metrics::MetricType;

    #[test]
    fn test_fit() {
        let csv_reader = CsvReader::new("./test.csv", (0, 1, 2), b'\t', false);
        let dataset = Dataset::new(csv_reader.into_iter());
        let mut nmf = Nmf::default();
        nmf.fit(&dataset);
    }

    #[test]
    fn test_predict() {
        let csv_reader = CsvReader::new("./test.csv", (0, 1, 2), b'\t', false);
        let dataset = Dataset::new(csv_reader.into_iter());
        let mut nmf = Nmf::default();
        nmf.fit(&dataset);
        let _pred = nmf.predict(
            dataset.user_to_idx[&dataset.users[0]],
            dataset.item_to_idx[&dataset.items[0]],
        );
    }

    #[test]
    fn test_evaluate() {
        let csv_reader = CsvReader::new("./test.csv", (0, 1, 2), b'\t', false);
        let dataset = Dataset::new(csv_reader.into_iter()).shuffle();
        let (trainset, testset) = dataset.train_test_split(80);

        let mut nmf = Nmf::default();
        nmf.fit(&trainset);
        let metrics = nmf.evaluate(
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
