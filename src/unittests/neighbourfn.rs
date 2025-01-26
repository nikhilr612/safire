use super::*;
use crate::lsops::random_perturbation;

#[test]
fn test_random_perturbation_dims() {
    let x = af::Array::new(&[1.0f32, 2.0, 3.0], af::Dim4::new(&[3, 1, 1, 1]));
    let perturbed = random_perturbation(&x, 0.1);
    assert_eq!(perturbed.dims(), x.dims());
}

#[test]
fn test_random_perturbation_scale() {
    let x = af::Array::new(&[1.0f32, 1.0, 1.0], af::Dim4::new(&[3, 1, 1, 1]));
    let scale = 0.0;
    let perturbed = random_perturbation(&x, scale);
    let mut result = vec![0.0f32; 3];
    perturbed.host(&mut result);
    assert_eq!(result, vec![1.0, 1.0, 1.0]);
}

#[test]
fn test_random_perturbation_different() {
    af::set_seed(0);
    let x = af::Array::new(&[1.0f32], af::Dim4::new(&[1, 1, 1, 1]));
    let scale = 1.0;
    let perturbed1 = random_perturbation(&x, scale);
    let perturbed2 = random_perturbation(&x, scale);
    let mut result1 = vec![0.0f32; 1];
    let mut result2 = vec![0.0f32; 1];
    perturbed1.host(&mut result1);
    perturbed2.host(&mut result2);
    assert_ne!(result1, result2);
}
