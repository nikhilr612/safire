//! Data-parallel simulated annealing.

use arrayfire::{self as af, dim4};

/// Performs data-parallel simulated annealing to minimize a numeric function.
///
/// # Type Parameters
///
/// * `E` - Function type that computes the energy/cost of a state. Must accept an `Array<f32>` and return an `Array<f32>`.
/// * `F` - Function type that generates neighboring states. Must accept an `Array<f32>` and return an `Array<f32>`.
/// * `G` - Iterator type that yields temperature values of type `f32` for the annealing schedule.
///
/// # Arguments
///
/// * `batch_size` - Number of parallel annealing chains to run
/// * `chain_length` - Number of iterations at each temperature
/// * `k` - Boltzmann constant used in acceptance probability calculation
/// * `start` - Initial state.
/// * `energy` - Function that computes the energy/cost of a state
/// * `neighbour_map` - Function that generates neighboring states
/// * `temperatures` - Iterator providing the temperature schedule
///
/// # Returns
///
/// The best state(s) found during the annealing process.
///
/// # Panics
///
/// Panics if the Boltzmann constant `k` is not positive (must be > 0.0)
pub fn minimize_numeric<E, F, G>(
    batch_size: u64,
    chain_length: usize,
    k: f32,
    start: &af::Array<f32>,
    energy: E,
    neighbour_map: F,
    temperatures: G,
) -> af::Array<f32>
where
    E: Fn(&af::Array<f32>) -> af::Array<f32>,
    F: Fn(&af::Array<f32>) -> af::Array<f32>,
    G: Iterator<Item = f32>,
{
    let tile_dim = dim4!(1, batch_size);
    let mut x = af::tile(start, tile_dim);
    let mut ex = energy(&x);

    assert!(k > 0.0, "Boltzmann constant must be positive");

    for temperature in temperatures {
        for _chain_idx in 0..chain_length {
            let n = neighbour_map(&x);
            let en = energy(&n);
            let logprobs = (&ex - &en) / (k * temperature);
            let diffs = af::gt(
                &af::exp(&logprobs),
                &af::randu::<f32>(dim4!(1, batch_size)),
                true,
            );
            x = af::select(&n, &diffs, &x);
            ex = af::select(&en, &diffs, &ex);
        }

        let (index, _min_energy) = af::imin(&ex, 1);
        let selected_xs = af::lookup(&x, &index, 1);
        x = af::tile(&selected_xs, tile_dim);
    }
    x
}
