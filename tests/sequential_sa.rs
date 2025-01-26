//! Unit tests for sequential simulated annealing.
use arrayfire as af;
use safire::{lsops::random_perturbation, seqsa, testfunctions};

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

#[test]
fn test_minimize_rastrigin() {
    af::set_seed(TEST_SEED);

    // Rastrigin function has global minimum at x = 0
    let energy = |x: &af::Array<f32>| {
        let result = testfunctions::rastrigin(x);
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
