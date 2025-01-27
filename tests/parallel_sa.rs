//! Unit tests for parallel simulated annealing.
use arrayfire as af;
use safire::{lsops::random_perturbation, parsa, testfunctions};

const TEST_SEED: u64 = 1737207124100;

// Reuse the same helper function for temperature schedule
fn exponential_schedule(start: f32, alpha: f32, steps: usize) -> impl Iterator<Item = f32> {
    let mut x = start;
    std::iter::from_fn(move || {
        let y = x;
        x *= alpha;
        Some(y)
    })
    .take(steps)
}

#[test]
fn test_minimize_rastrigin() {
    af::set_seed(TEST_SEED);

    let start = af::constant(1.0f32, af::dim4!(2, 1)); // 2D starting point

    let result = parsa::minimize_numeric(
        800,  // batch size
        10,   // chain length
        0.01, // Boltzmann constant
        &start,
        testfunctions::rastrigin,
        |x| random_perturbation(x, 0.4),
        exponential_schedule(800.0, 0.8, 20),
    );

    let mut host_result = vec![0.0f32; 2 * 800];
    result.host(&mut host_result);
    host_result.truncate(4);

    // Check if result is close to global minimum (0,0)
    assert!(
        host_result.iter().all(|&x| x.abs() < 0.1),
        "Expected x_i = 0, got {host_result:?}"
    );
}

#[test]
fn test_minimize_ackley() {
    af::set_seed(TEST_SEED);

    let start = af::constant(1.0f32, af::dim4!(3, 1)); // 3D starting point

    let result = parsa::minimize_numeric(
        100,
        100,
        0.001,
        &start,
        testfunctions::ackley,
        |x| random_perturbation(x, 0.2),
        exponential_schedule(500.0, 0.8, 20),
    );

    let mut host_result = vec![0.0f32; 3 * 100];
    result.host(&mut host_result);
    host_result.truncate(6);

    // Check if result is close to global minimum (0,0,0)
    assert!(
        host_result.iter().all(|&x| x.abs() < 0.11),
        "Expected x_i = 0, got {host_result:?}"
    );
}

#[test]
fn test_minimize_schwefel_small() {
    af::set_seed(TEST_SEED);

    let start = af::constant(380.0f32, af::dim4!(2, 1)); // 2D starting point

    let result = parsa::minimize_numeric(
        100,
        20,
        0.01,
        &start,
        testfunctions::schwefel,
        |x| random_perturbation(x, 8.0),
        exponential_schedule(600.0, 0.75, 15),
    );

    let mut host_result = vec![0.0f32; 2 * 100];
    result.host(&mut host_result);
    host_result.truncate(4);

    // Check if result is close to global minimum (≈420.9687)
    const EXPECTED: f32 = 420.9687;
    assert!(
        host_result.iter().all(|&x| (x - EXPECTED).abs() < 1.0),
        "Expected x_i = {EXPECTED}, got: {host_result:?}"
    );
}

#[test]
#[should_panic(expected = "Boltzmann constant must be positive")]
fn test_invalid_boltzmann_constant() {
    let start = af::constant(1.0f32, af::dim4!(2, 1));
    let neighbour = |x: &af::Array<f32>| random_perturbation(x, 0.2);

    parsa::minimize_numeric(
        10,
        100,
        0.0, // Invalid k
        &start,
        testfunctions::ackley,
        neighbour,
        exponential_schedule(1000.0, 0.8, 20),
    );
}
