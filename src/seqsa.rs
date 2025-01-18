//! Sequential Simulated Annealing.

use tinyrand::{Probability, Rand, Seeded, StdRand};

fn cpu_rand() -> tinyrand::StdRand {
    StdRand::seed(1737207124100)
}

/// Minimize an objective function through sequential simulated annealing.
/// It works by iteratively exploring the solution space while gradually
/// "cooling" the system according to a temperature schedule.
///
/// # Arguments
///
/// * `batch_size` - Number of iterations to perform at each temperature
/// * `k` - Boltzmann constant that scales the acceptance probability
/// * `start` - Initial state/solution
/// * `energy` - Objective function that evaluates the "energy" (cost) of a state
/// * `neighbour` - Function that randomly picks a neighboring state from the current one
/// * `temperatures` - Iterator providing the cooling schedule temperatures
///
/// # Type Parameters
///
/// * `T` - Type representing a state/solution in the search space
/// * `E` - Type of the energy function `Fn(&T) -> f32`
/// * `F` - Type of the neighbor function `Fn(&T) -> T`
/// * `G` - Type of the temperature iterator `Iterator<Item = f32>`
pub fn minimize<T, E, F, G>(
    batch_size: usize,
    k: f32,
    start: T,
    energy: E,
    neighbour: F,
    temperatures: G,
    random_seed: u64,
) -> T
where
    E: Fn(&T) -> f32,
    F: Fn(&T) -> T,
    G: Iterator<Item = f32>,
{
    let mut x = start;
    let mut ex = energy(&x);
    let mut rand = StdRand::seed(random_seed);

    assert!(k > 0.0, "Boltzmann constant must be positive");

    for temperature in temperatures {
        if temperature == 0.0 {
            break;
        }

        for _ in 0..batch_size {
            let n = neighbour(&x);
            let en = energy(&n);

            if en.is_nan() {
                continue;
            }

            if en < ex {
                x = n;
                ex = en;
                continue;
            }

            let p = f64::exp(f64::from((ex - en) / (k * temperature)));
            if rand.next_bool(Probability::new(p)) {
                x = n;
                ex = en;
            }
        }
    }

    x
}
