pub fn rmse(preds: &[f64], actuals: &[f64]) -> f64 {
    let sum: f64 = preds
        .iter()
        .zip(actuals.iter())
        .map(|(&p, &a)| (p - a) * (p - a))
        .sum();

    (sum / preds.len() as f64).sqrt()
}

pub fn mae(preds: &[f64], actuals: &[f64]) -> f64 {
    let sum: f64 = preds
        .iter()
        .zip(actuals.iter())
        .map(|(&p, &a)| (p - a).abs())
        .sum();

    sum / preds.len() as f64
}

pub fn mape(preds: &[f64], actuals: &[f64]) -> f64 {
    let sum: f64 = preds
        .iter()
        .zip(actuals.iter())
        .map(|(&p, &a)| ((a - p) / (a + 1e-9)).abs())
        .sum();

    100.0 * sum / preds.len() as f64
}

pub fn smape(preds: &[f64], actuals: &[f64]) -> f64 {
    let sum: f64 = preds
        .iter()
        .zip(actuals.iter())
        .map(|(&p, &a)| 2.0 * (p - a).abs() / (p.abs() + a.abs() + 1e-9))
        .sum();

    100.0 * sum / preds.len() as f64
}

pub enum MetricType {
    Rmse,
    Mae,
    Mape,
    Smape,
}

pub enum MetricValue {
    Rmse(f64),
    Mae(f64),
    Mape(f64),
    Smape(f64),
}
