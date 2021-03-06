// Adapted from https://github.com/NicolasHug/Surprise/blob/master/surprise/prediction_algorithms/knns.py
use crate::algos::Algorithm;
use crate::data::Dataset;
use crate::similarities::{cosine, msd, Similarity};
use na::{DMatrix, DVector};
use ordered_float::OrderedFloat;
use std::collections::{BinaryHeap, HashMap};

pub struct Knn {
    pub k: usize,
    pub min_k: usize,
    pub user_based: bool,
    pub similarity: Similarity,
    sim: Option<DMatrix<f64>>,
    means: Option<DVector<f64>>,
    edges: HashMap<usize, Vec<(usize, f64)>>,
}

impl Knn {
    pub fn new(k: usize, min_k: usize, user_based: bool, similarity: Similarity) -> Self {
        Self {
            k,
            min_k,
            user_based,
            similarity,
            sim: None,
            means: None,
            edges: HashMap::new(),
        }
    }

    fn compute_similarities(&mut self, size: usize, dataset: &Dataset) -> DMatrix<f64> {
        let mut matrix = DMatrix::zeros(dataset.num_users, dataset.num_items);

        for key in 0..size {
            self.edges.insert(key, vec![]);
        }

        for i in 0..dataset.len() {
            let user = dataset.user_to_idx[&dataset.users[i]];
            let item = dataset.item_to_idx[&dataset.items[i]];
            let value = dataset.values[i];
            matrix[(user, item)] = value;

            if self.user_based {
                self.edges.get_mut(&item).unwrap().push((user, value));
            } else {
                self.edges.get_mut(&user).unwrap().push((item, value));
            }
        }

        if !self.user_based {
            matrix = matrix.transpose();
        }

        match self.similarity {
            Similarity::Cosine => cosine(matrix),
            Similarity::Msd => msd(matrix),
        }
    }
}

impl Algorithm for Knn {
    fn fit(&mut self, dataset: &Dataset) {
        let size = if self.user_based {
            dataset.num_items
        } else {
            dataset.num_users
        };

        self.sim = Some(self.compute_similarities(size, dataset));

        let mut sums = DVector::zeros(size);
        let mut counts = DVector::zeros(size);

        for i in 0..dataset.len() {
            let user = dataset.user_to_idx[&dataset.users[i]];
            let value = dataset.values[i];
            if self.user_based {
                sums[user] += value;
                counts[user] += 1.0;
            }
        }

        self.means = Some(sums.component_div(&counts));
    }

    fn predict(&self, user_idx: usize, item_idx: usize) -> f64 {
        let sim = self.sim.as_ref().unwrap();

        let (x1, y) = if self.user_based {
            (user_idx, item_idx)
        } else {
            (item_idx, user_idx)
        };

        let mut neighbors: BinaryHeap<_> = self.edges[&y]
            .iter()
            .map(|&(x2, value)| (x2, OrderedFloat(sim[(x1, x2)]), OrderedFloat(value)))
            .collect();

        let means = self.means.as_ref().unwrap();
        let est = means[x1];

        let mut sum_sim = 0.0;
        let mut sum_sim_times_val = 0.0;
        let mut k = 0;

        while let Some((x2, s, value)) = neighbors.pop() {
            sum_sim += s.0;
            sum_sim_times_val += s.0 * (value.0 - means[x2]);
            k += 1;

            if k == self.k {
                break;
            }
        }

        if k < self.min_k {
            sum_sim = 0.0;
        }

        if sum_sim_times_val == 0.0 {
            est
        } else {
            est + sum_sim / sum_sim_times_val
        }
    }
}

impl Default for Knn {
    fn default() -> Self {
        Self::new(40, 1, true, Similarity::Cosine)
    }
}

#[cfg(test)]
mod tests {
    use crate::algos::{Algorithm, Evaluate, Knn};
    use crate::data::{CsvReader, Dataset};
    use crate::metrics::MetricType;

    #[test]
    fn test_fit() {
        let csv_reader = CsvReader::new("./test.csv", (0, 1, 2), b'\t', false);
        let dataset = Dataset::new(csv_reader.into_iter());
        let mut knn = Knn::default();
        knn.fit(&dataset);
    }

    #[test]
    fn test_predict() {
        let csv_reader = CsvReader::new("./test.csv", (0, 1, 2), b'\t', false);
        let dataset = Dataset::new(csv_reader.into_iter());
        let mut knn = Knn::default();
        knn.fit(&dataset);
        let _pred = knn.predict(
            dataset.user_to_idx[&dataset.users[0]],
            dataset.item_to_idx[&dataset.items[0]],
        );
    }

    #[test]
    fn test_evaluate() {
        let csv_reader = CsvReader::new("./test.csv", (0, 1, 2), b'\t', false);
        let dataset = Dataset::new(csv_reader.into_iter()).shuffle();
        let (trainset, testset) = dataset.train_test_split(0.8);

        let mut knn = Knn::default();
        knn.fit(&trainset);
        let metrics = knn.evaluate(
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
