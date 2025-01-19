use crate::testfunctions;
use arrayfire as af;

macro_rules! assert_float_eq {
    ($a: expr, $b: expr) => {
        assert_float_eq!($a, $b, 1e-5);
    };

    ($a: expr, $b: expr, $error: literal) => {
        let a = $a;
        let b = $b;
        let abs_error = (a - b).abs();
        assert!(
            abs_error < $error,
            "Expected {b}, got {a} (error = {abs_error} > {})",
            $error
        );
    };
}

fn to_scalar(x: af::Array<f32>) -> f32 {
    let mut host_array = [0.0];
    x.host(&mut host_array);
    host_array[0]
}

#[cfg(test)]
mod objectivefn;
