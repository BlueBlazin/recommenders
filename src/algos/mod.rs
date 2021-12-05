mod knn;
mod nmf;
mod svd;

pub use crate::algos::knn::Knn;
pub use crate::algos::nmf::Nmf;
pub use crate::algos::svd::Svd;
use crate::data::Dataset;
use crate::metrics::*;

pub trait Algorithm {
    fn fit(&mut self, dataset: &Dataset);
    fn predict(&self, user_idx: usize, item_idx: usize) -> f64;

    fn predict_iter<T: Iterator<Item = (usize, usize)>>(
        &self,
        data: T,
    ) -> Vec<(usize, usize, f64)> {
        data.map(|(u_idx, i_idx)| (u_idx, i_idx, self.predict(u_idx, i_idx)))
            .collect()
    }
}

pub trait Evaluate {
    fn evaluate(&self, dataset: &Dataset, metrics: Vec<MetricType>) -> Vec<MetricValue>;
}

impl<T> Evaluate for T
where
    T: Algorithm,
{
    fn evaluate(&self, dataset: &Dataset, metrics: Vec<MetricType>) -> Vec<MetricValue> {
        let preds: Vec<_> = (0..dataset.len())
            .map(|i| {
                let user_idx = dataset.user_to_idx[&dataset.users[i]];
                let item_idx = dataset.item_to_idx[&dataset.items[i]];
                self.predict(user_idx, item_idx)
            })
            .collect();

        let actuals: Vec<_> = (0..dataset.len()).map(|i| dataset.values[i]).collect();
        let mut metric_vals = vec![];

        for metric in metrics {
            match metric {
                MetricType::Rmse => metric_vals.push(MetricValue::Rmse(rmse(&preds, &actuals))),
                MetricType::Mae => metric_vals.push(MetricValue::Mae(mae(&preds, &actuals))),
                MetricType::Mape => metric_vals.push(MetricValue::Mape(mape(&preds, &actuals))),
                MetricType::Smape => metric_vals.push(MetricValue::Smape(smape(&preds, &actuals))),
            }
        }

        metric_vals
    }
}
