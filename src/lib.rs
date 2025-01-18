//! A small library for simulated annealing using arrayfire.

mod seqsa;
mod testfunctions;

use arrayfire::{self as af};

macro_rules! assert_float_eq {
    ($a: expr, $b: expr) => {
        assert_float_eq!($a, $b, 1e-5);
    };

    ($a: expr, $b: expr, $error: literal) => {
        let a = $a;
        let b = $b;
        let abs_error = (a - b).abs();
        assert!(
            abs_error < $error,
            "Expected {b}, got {a} (error = {abs_error} > {})",
            $error
        );
    };
}

fn to_scalar(x: af::Array<f32>) -> f32 {
    let mut host_array = [0.0];
    x.host(&mut host_array);
    host_array[0]
}

#[cfg(test)]
mod objectivefn_tests {
    use super::*;

    #[test]
    fn test_rastrigin_global_minimum() {
        // Test global minimum at x = 0
        let x = af::constant(0.0f32, af::dim4!(2, 1));
        let result = testfunctions::rastrigin(&x);

        let host_result = to_scalar(result);
        assert_float_eq!(host_result, 0.0);
    }

    #[test]
    fn test_rastrigin_single_dimension() {
        // Test with a single point x = 1.0
        let x = af::constant(1.0f32, af::dim4!(1, 1));
        let result = testfunctions::rastrigin(&x);

        let host_result = to_scalar(result);
        // f(1) = 10 * 1 + (1² - 10*cos(2π*1)) ≈ 1
        assert_float_eq!(host_result, 1.0);
    }

    #[test]
    fn test_rastrigin_multiple_dimensions() {
        // Test with 2D input [1.0, 2.0]
        let input = vec![1.0f32, 2.0f32];
        let x = af::Array::new(&input, af::dim4!(2, 1));
        let result = testfunctions::rastrigin(&x);

        let host_result = to_scalar(result);
        // f([1,2]) = 10*2 + (1² - 10*cos(2π*1) + 2² - 10*cos(2π*2)) ≈ 5
        assert_float_eq!(host_result, 5.0);
    }

    #[test]
    fn test_rastrigin_multiple_points() {
        // Test with multiple points: two 2D points
        let input = vec![0.0f32, 0.0f32, 1.0f32, 1.0f32];
        let x = af::Array::new(&input, af::dim4!(2, 2));
        let result = testfunctions::rastrigin(&x);

        let mut host_result = [0.0, 0.0];
        result.host(&mut host_result);

        // First point [0,0] should give 0
        assert_float_eq!(host_result[0], 0.0f32);
        // Second point [1,1] should give ≈ 2
        assert_float_eq!(host_result[1], 2.0f32);
    }

    #[test]
    fn test_rastrigin_symmetry() {
        // Test symmetry: f(x) = f(-x)
        let x_pos = af::Array::new(&[1.0f32], af::dim4!(1, 1));
        let x_neg = af::Array::new(&[-1.0f32], af::dim4!(1, 1));

        let result_pos = testfunctions::rastrigin(&x_pos);
        let result_neg = testfunctions::rastrigin(&x_neg);

        let host_pos = to_scalar(result_pos);
        let host_neg = to_scalar(result_neg);

        assert!(
            (host_neg - host_pos).abs() < 1e-5,
            "Implementation is not symmetric, although funciton is."
        );
    }

    #[test]
    fn test_ackley_zero() {
        // Test minimum at x = 0
        let x = af::constant(0.0f32, af::dim4!(1));
        let result = to_scalar(testfunctions::ackley(&x));
        assert_float_eq!(result, 0.0);
    }

    #[test]
    fn test_ackley_one() {
        // Test function value at x = 1
        let x = af::constant(1.0f32, af::dim4!(1));
        let result = to_scalar(testfunctions::ackley(&x));
        let expected = 3.62538f32; // Pre-calculated value
        assert_float_eq!(result, expected);
    }

    #[test]
    fn test_ackley_vector() {
        // Test with vector input
        let x = af::Array::new(&[1.0f32, 2.0f32, 3.0f32], af::Dim4::new(&[3, 1, 1, 1]));
        let result = testfunctions::ackley(&x);
        assert!(result.elements() == 1);

        let v = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let x = af::Array::new(&v, af::dim4!(3, 2));
        let result = testfunctions::ackley(&x);
        let mut host_result = [0.0, 0.0];
        result.host(&mut host_result);

        assert_float_eq!(host_result[0], 0.0f32);
        assert_float_eq!(host_result[1], 3.62538f32);
    }

    #[test]
    fn test_schwefel_global_minimum() {
        // Test at global minimum x ≈ 420.9687
        let x = af::constant(420.9687f32, af::dim4!(1, 1));
        let result = testfunctions::schwefel(&x);
        let host_result = to_scalar(result);
        assert_float_eq!(host_result, 0.0);
    }

    #[test]
    fn test_schwefel_multiple_dimensions() {
        // Test with 2D input at global minimum
        let input = vec![420.9687f32, 420.9687f32];
        let x = af::Array::new(&input, af::dim4!(2, 1));
        let result = testfunctions::schwefel(&x);
        let host_result = to_scalar(result);
        assert_float_eq!(host_result, 0.0);
    }

    #[test]
    fn test_schwefel_multiple_points() {
        // Test with two points: one at global minimum, one at origin
        let input = vec![420.9687f32, 420.9687f32, 0.0f32, 0.0f32];
        let x = af::Array::new(&input, af::dim4!(2, 2));
        let result = testfunctions::schwefel(&x);

        let mut host_result = [0.0, 0.0];
        result.host(&mut host_result);

        // First point [420.9687, 420.9687] should give approximately 0
        assert_float_eq!(host_result[0], 0.0f32);
        // Second point [0, 0] should give approximately 837.9658 (2 * 418.9829)
        assert_float_eq!(host_result[1], 837.9658);
    }

    #[test]
    fn test_schwefel_origin() {
        // Test function value at origin
        let x = af::constant(0.0f32, af::dim4!(1, 1));
        let result = testfunctions::schwefel(&x);
        let host_result = to_scalar(result);
        assert_float_eq!(host_result, 418.9829);
    }

    #[test]
    fn test_schwefel_dimensionality() {
        // Test that the function scales correctly with dimensionality
        let x = af::constant(0.0f32, af::dim4!(3, 1));
        let result = testfunctions::schwefel(&x);
        let host_result = to_scalar(result);
        // At origin, value should be A*n where A=418.9829 and n=3
        assert_float_eq!(host_result, 3.0 * 418.9829);
    }
}

#[cfg(test)]
mod seq_sa_tests {
    use super::*;
    use arrayfire as af;

    const TEST_SEED: u64 = 1737207124100;

    // Helper function to create an exponential schedule.
    fn exponential_schedule(start: f32, alpha: f32, steps: usize) -> impl Iterator<Item = f32> {
        let mut x = start;
        std::iter::from_fn(move || {
            let y = x;
            x *= alpha;
            Some(y)
        })
        .take(steps)
    }

    fn random_perturbation(x: &af::Array<f32>, scale: f32) -> af::Array<f32> {
        let dims = x.dims();
        let noise = af::randn::<f32>(dims) * scale;
        x + noise
    }

    #[test]
    fn test_minimize_rastrigin() {
        af::set_seed(TEST_SEED);

        // Rastrigin function has global minimum at x = 0
        let energy = |x: &af::Array<f32>| {
            let result = crate::testfunctions::rastrigin(x);
            let mut host_val = [0.0f32];
            result.host(&mut host_val);
            host_val[0]
        };

        let neighbour = |x: &af::Array<f32>| random_perturbation(x, 0.2);

        let start = af::constant(1.0f32, af::dim4!(2, 1)); // 2D starting point

        let result = seqsa::minimize(
            800, // batch size
            0.1, // Boltzmann constant
            start,
            energy,
            neighbour,
            exponential_schedule(1000.0, 0.8, 25),
            TEST_SEED,
        );

        dbg!(energy(&result));

        let mut host_result = vec![0.0f32; 2];
        result.host(&mut host_result);

        // Check if result is close to global minimum (0,0)
        assert!(
            host_result.iter().all(|&x| x.abs() < 0.1),
            "Expected x_i = 0, got {host_result:?}"
        );
    }

    #[test]
    fn test_minimize_ackley() {
        af::set_seed(TEST_SEED);

        // Ackley function has global minimum at x = 0
        let energy = |x: &af::Array<f32>| {
            let result = crate::testfunctions::ackley(x);
            let mut host_val = [0.0f32];
            result.host(&mut host_val);
            host_val[0]
        };

        let neighbour = |x: &af::Array<f32>| random_perturbation(x, 0.2);

        let start = af::constant(1.0f32, af::dim4!(3, 1)); // 3D starting point

        let result = seqsa::minimize(
            1000,
            0.001,
            start,
            energy,
            neighbour,
            exponential_schedule(1000.0, 0.8, 20),
            TEST_SEED,
        );

        let mut host_result = vec![0.0f32; 3];
        result.host(&mut host_result);

        // Check if result is close to global minimum (0,0,0)
        assert!(
            host_result.iter().all(|&x| x.abs() < 0.11),
            "Expected x_i = 0, got {host_result:?}"
        );
    }

    #[test]
    fn test_minimize_schwefel() {
        af::set_seed(TEST_SEED);

        // Schwefel function has global minimum at x ≈ 420.9687
        let energy = |x: &af::Array<f32>| {
            let result = crate::testfunctions::schwefel(x);
            let mut host_val = [0.0f32];
            result.host(&mut host_val);
            host_val[0]
        };

        let neighbour = |x: &af::Array<f32>| random_perturbation(x, 8.0);

        let start = af::constant(380.0f32, af::dim4!(2, 1)); // 2D starting point

        let result = seqsa::minimize(
            300,
            0.01,
            start,
            energy,
            neighbour,
            exponential_schedule(1000.0, 0.8, 30),
            TEST_SEED,
        );

        let mut host_result = vec![0.0f32; 2];
        result.host(&mut host_result);

        // Check if result is close to global minimum (≈420.9687)
        const EXPECTED: f32 = 420.9687;
        assert!(
            host_result.iter().all(|&x| (x - EXPECTED).abs() < 1.0),
            "Expected x_i = {EXPECTED}, got: {host_result:?}"
        );
    }
}
