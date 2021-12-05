use na::DMatrix;

pub fn cosine(matrix: DMatrix<f64>) -> DMatrix<f64> {
    let norms = (&matrix.component_mul(&matrix)).column_sum().map(f64::sqrt);
    (&matrix * &matrix.transpose()).component_div(&(&norms * &norms.transpose()))
}

pub fn msd(matrix: DMatrix<f64>) -> DMatrix<f64> {
    let (m, n) = matrix.shape();
    let mut sq_diff: DMatrix<f64> = DMatrix::zeros(m, m);
    let mut freq: DMatrix<f64> = DMatrix::zeros(m, m);
    let mut sim = DMatrix::zeros(m, m);

    for u in 0..m {
        for v in 0..m {
            for w in 0..n {
                let ru = matrix[(u, w)];
                let rv = matrix[(v, w)];
                if ru != 0.0 && rv != 0.0 {
                    sq_diff[(u, v)] += (ru - rv) * (ru - rv);
                    freq[(u, v)] += 1.0;
                }
            }
        }
    }

    for u in 0..m {
        sim[(u, u)] = 1.0;
        for v in u + 1..m {
            if freq[(u, v)] == 0.0 {
                sim[(u, v)] = 0.0;
            } else {
                sim[(u, v)] = 1.0 / (sq_diff[(u, v)] / freq[(u, v)] + 1.0);
            }

            sim[(v, u)] = sim[(u, v)];
        }
    }

    sim
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

    #[test]
    fn test_msd_similarity() {
        let matrix = DMatrix::from_vec(
            3,
            4,
            vec![0.0, 1.0, 0.0, 3.0, 0.0, 0.0, 1.0, 2.0, 4.0, 0.0, 2.0, 0.0],
        );

        let expected = DMatrix::from_vec(3, 3, vec![1.0, 0.5, 0.1, 0.5, 1.0, 0.2, 0.1, 0.2, 1.0]);

        let sim = msd(matrix);
        assert!(sim.relative_eq(&expected, 1e-7, 1e-7));
    }
}
