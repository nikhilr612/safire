# SaFire 🔥

SAFire is a Rust library that implements simulated annealing optimization algorithms using ArrayFire for efficient computation, especially on GPUs.

## Overview

Safire provides both sequential and data-parallel implementations of simulated annealing, making it suitable for both small-scale and large-scale optimization problems. The library leverages ArrayFire's GPU acceleration capabilities to perform efficient parallel computations.

### Features

- ✅ Sequential simulated annealing implementation
- ✅ Synchronous data-parallel simulated annealing
- 📊 Common benchmark functions included:
  - Rastrigin function
  - Ackley function
  - Schwefel function
- 🔧 Flexible API for custom optimization problems
- 🚀 GPU acceleration support via ArrayFire
- 🔄 Customizable temperature schedules
- 🎯 Automatic parameter validation

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
safire = { git = "https://github.com/nikhilr612/safire" }
```

## Usage

```rust
use safire::{parsa, lsops::random_perturbation, testfunctions};
use arrayfire as af;

fn main() {
    let start = af::constant(1.0f32, af::dim4!(2, 1));
    let schedule = (0..20).map(|i| 800.0 * 0.8f32.powi(i));

    // Perform synchronous data-parallel simulated annealing.
    // For sequential counter-part see [`safire::seqsa::minimize`].
    let result = parsa::minimize_numeric(
        800,                    // batch size
        10,                     // chain length
        0.01,                   // Boltzmann constant
        &start,                 // starting point
        testfunctions::rastrigin, // objective function
        |x| random_perturbation(x, 0.4), // neighbor function
        schedule,
    );

    // `result` now has an array of solution candidates.
    // ... use result ...
}
```

> **Note**: The first run of GPU-enabled functions may be slower due to JIT compilation and device initialization by ArrayFire. Subsequent runs will be significantly faster.

## Benchmark Functions

Safire includes several standard test functions for benchmarking optimization algorithms:

- **Rastrigin Function**: Non-convex function with many local minima (global minimum at x = 0)
- **Ackley Function**: Continuous, multimodal function (global minimum at x = 0)
- **Schwefel Function**: Complex function with many local minima (global minimum at x ≈ 420.9687)

## Implementation Details

### Sequential Simulated Annealing
- Traditional implementation suitable for single-solution optimization
- Flexible temperature scheduling
- Customizable neighbor generation and energy functions
- Seeded random number generation for reproducibility

### Data-Parallel Simulated Annealing
- Synchronous parallel implementation optimized for GPU execution
- Efficient batch processing of multiple chains simultaneously
- Configurable chain length and batch size
- Direct GPU memory management via ArrayFire

## Dependencies

- [ArrayFire](https://github.com/arrayfire/arrayfire): For efficient array operations and GPU computing

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References
1. Ferreiro, A.M. et al. (2012) 'An efficient implementation of parallel simulated annealing algorithm in gpus', Journal of Global Optimization, 57(3), pp. 863–890. doi:10.1007/s10898-012-9979-z.
2. Kirkpatrick, S., Gelatt, C.D. and Vecchi, M.P. (1983) 'Optimization by simulated annealing', Science, 220(4598), pp. 671–680. doi:10.1126/science.220.4598.671.
3. Czech, A., & Wieloch, B. (2016). Data-parallel simulated annealing using graphics processing units. Journal of Parallel and Distributed Computing.
