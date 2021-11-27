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
