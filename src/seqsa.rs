//! Sequential Simulated Annealing.

use tinyrand::{Probability, Rand, Seeded, StdRand};

/// Minimize an objective function through sequential simulated annealing.
/// It works by iteratively exploring the solution space while gradually
/// "cooling" the system according to a temperature schedule.
///
/// # Arguments
///
/// * `chain_length` - Number of iterations to perform at each temperature
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
///
/// # Panics
///
/// Panics if the Boltzmann constant `k` is not positive.
pub fn minimize<T, E, F, G>(
    chain_length: usize,
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

        for _ in 0..chain_length {
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

/// Minimize an objective function through sequential simulated annealing,
/// returning an iterator that yields solutions at each temperature step.
///
/// Similar to [`minimize`], but returns an iterator instead of a single solution,
/// allowing for monitoring the optimization process. The iterator yields the current
/// best solution after processing each temperature in the cooling schedule.
///
/// # Arguments
///
/// * `chain_length` - Number of iterations to perform at each temperature
/// * `k` - Boltzmann constant that scales the acceptance probability
/// * `start` - Initial state/solution
/// * `energy` - Objective function that evaluates the "energy" (cost) of a state
/// * `neighbour` - Function that randomly picks a neighboring state from the current one
/// * `temperatures` - Iterator providing the cooling schedule temperatures
/// * `random_seed` - Seed for the random number generator
///
/// # Type Parameters
///
/// * `T` - Type representing a state/solution in the search space, must implement Clone
///         to allow copying solutions between iterations
/// * `E` - Type of the energy function `Fn(&T) -> f32`, must be callable and 'iter-lifetime bounded
/// * `F` - Type of the neighbor function `Fn(&T) -> T`, must be callable and 'iter-lifetime bounded
/// * `G` - Type of the temperature iterator `Iterator<Item = f32>`, must implement Iterator
///         and be 'iter-lifetime bounded
///
/// # Panics
///
/// Panics if the Boltzmann constant `k` is not positive.
pub fn minimize_lazy<'iter, T, E, F, G>(
    chain_length: usize,
    k: f32,
    start: T,
    energy: E,
    neighbour: F,
    temperatures: G,
    random_seed: u64,
) -> impl Iterator<Item = T> + 'iter
where
    T: Clone + 'iter,
    E: Fn(&T) -> f32 + 'iter,
    F: Fn(&T) -> T + 'iter,
    G: Iterator<Item = f32> + 'iter,
{
    let mut x = start;
    let mut ex = energy(&x);
    let mut rand = StdRand::seed(random_seed);

    assert!(k > 0.0, "Boltzmann constant must be positive");

    temperatures
        .take_while(|&t| t > 0.0)
        .map(move |temperature| {
            for _ in 0..chain_length {
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
            x.clone()
        })
}
