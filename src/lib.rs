//! A small library for simulated annealing using arrayfire.

#[warn(clippy::pedantic)]
// Public APIs
pub mod seqsa;
pub mod testfunctions;

#[cfg(test)]
// Unit tests.
mod unittests;

// Re-export arrayfire.
pub use arrayfire::{self as af};
