//!
//! Module to implement local search operations.
//! Local Search Operators refer to operators, or functions on search points (candidate solutions)
//! used to sample random candidate solutions related to a given solution in Simulated Annealing.

use arrayfire::{self as af};

/// Creates a perturbed version of an input vector by adding random Gaussian noise scaled by the given factor.
/// Returns a new array with random noise added to the input.
/// Useful as a local search for numeric minimization problems.
#[must_use]
pub fn random_perturbation(x: &af::Array<f32>, scale: f32) -> af::Array<f32> {
    let dims = x.dims();
    let noise = af::randn::<f32>(dims) * scale;
    x + noise
}

// TODO: Implement Swap operator.
// fn random_swap(x: &af::Array<u32>) -> af::Array<u32> {
//     let n = x.dims()[1]; // How many sequences to randomly swap in parallel.
//     let l = x.dims()[0]; // What is the length of each sequence.

//     todo!("Implement swapping..")
// }
