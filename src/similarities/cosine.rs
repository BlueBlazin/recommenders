use na::DMatrix;

pub fn cosine(matrix: DMatrix<f64>) -> DMatrix<f64> {
    let norms = (&matrix.component_mul(&matrix)).column_sum().map(f64::sqrt);
    (&matrix * &matrix.transpose()).component_div(&(&norms * &norms.transpose()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let matrix = DMatrix::from_vec(
            3,
            4,
            vec![0.0, 1.0, 0.0, 3.0, 0.0, 0.0, 1.0, 2.0, 4.0, 0.0, 2.0, 0.0],
        );

        let expected = DMatrix::from_vec(
            3,
            3,
            vec![
                1.0, 0.21081851, 0.31622777, 0.21081851, 1.0, 0.66666667, 0.31622777, 0.66666667,
                1.0,
            ],
        );

        let sim = cosine(matrix);
        assert!(sim.relative_eq(&expected, 1e-7, 1e-7));
    }
}
