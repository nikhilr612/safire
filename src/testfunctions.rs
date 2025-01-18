//! Implement test functions for use as objectives in optmization.

use std::f32::consts::{E, PI};

use arrayfire as af;

/// The Ackley function is a continuous, non-convex and widely used for testing
/// optimization algorithms. It has a global minimum of 0 at x = 0.
/// Mathematically,
/// ```other
/// f(x) = -20 * exp(-0.2 * sqrt(1/n * sum(x_i^2))) - exp(1/n * sum(cos(2π*x_i))) + 20 + e
/// ```
/// # Parameters
/// - x: Input array of values to evaluate. The first dimension specifies the number of `x_i` for `f(x)`.
/// So, an input array of dim4(3,2) is will evaluate the ackley funciton on two 3d vectors.
///
/// # Returns
/// - Array containing the Ackley function value applied along the first dimension.
///
pub fn ackley(x: &af::Array<f32>) -> af::Array<f32> {
    const A: f32 = 20.0;
    const B: f32 = 0.2;
    const C: f32 = 2.0 * PI;

    let x2 = x * x;
    let n = x.dims()[0] as f32;

    let rmx2 = af::sqrt(&(af::sum(&x2, 0) / n));

    let mcosx = af::sum(&af::cos(&(C * x)), 0) / n;

    // Formula
    -A * af::exp(&(-B * rmx2)) - af::exp(&mcosx) + A + E
}

/// The Rastrigin function is a non-convex function used as a performance test problem for optimization algorithms.
/// It has a global minimum of 0 at x = 0 and many local minima.
/// Mathematically,
/// ```other
/// f(x) = An + sum(x_i^2 - A * cos(2π*x_i))
/// ```
/// where A = 10 and n is the dimension of x.
///
/// # Parameters
/// - x: Input array of values to evaluate. The first dimension specifies the number of `x_i` for `f(x)`.
///
/// # Returns
/// - Array containing the Rastrigin function value applied along the first dimension.
///
pub fn rastrigin(x: &af::Array<f32>) -> af::Array<f32> {
    const A: f32 = 10.0;
    let n = x.dims()[0] as f32;
    let v = x * x - A * af::cos(&(2.0 * PI * x));
    A * n + af::sum(&v, 0)
}

/// The Schwefel function is a continuous, multimodal function used as a benchmark for optimization algorithms.
/// It has a global minimum of 0 at x = 420.9687 and many local minima.
/// Mathematically,
/// ```other
/// f(x) = 418.9829n - sum(x_i * sin(sqrt(|x_i|)))
/// ```
/// where n is the dimension of x.
///
/// # Parameters
/// - x: Input array of values to evaluate. The first dimension specifies the number of `x_i` for `f(x)`.
///
/// # Returns
/// - Array containing the Schwefel function value applied along the first dimension.
///
pub fn schwefel(x: &af::Array<f32>) -> af::Array<f32> {
    const A: f32 = 418.9829;
    let n = x.dims()[0] as f32;

    let v = x * af::sin(&af::sqrt(&x));
    A * n - af::sum(&v, 0)
}
