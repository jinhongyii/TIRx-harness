//! NumPy-backed `C = A @ B^T` (feature `numpy`).
//!
//! Legacy source: `engine-rs/src/numpy_backend.rs`, module `python`
//! (`F32Matrix`, `NumpyBackendError`, `matmul_f32_abt`, `with_numpy`,
//! `numpy_f32_matrix`, `owned_rust_f32_values`) and its `python_tests`.
//! `ProfileTimer` instrumentation was dropped.
//!
//! The result is whatever `numpy.matmul` (the linked BLAS sgemm) produces: the
//! summation order is unspecified and may differ between NumPy builds, CPUs
//! and thread counts. Use it only where the legacy engine did (the
//! unobserved canonical BF16 tile GEMM); see the contract in `crate::mma`.

use std::error::Error;
use std::fmt;

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyDictMethods, PyModule};

use super::backend::MatmulBackend;
use crate::types::{OpError, OpResult};

/// A dense row-major matrix owned entirely by Rust.
#[derive(Clone, Debug, PartialEq)]
pub struct F32Matrix {
    rows: usize,
    cols: usize,
    values: Vec<f32>,
}

impl F32Matrix {
    /// Row-major `rows x cols` f32 matrix; errors when `values.len() != rows * cols` or the
    /// product overflows.
    pub fn new(rows: usize, cols: usize, values: Vec<f32>) -> Result<Self, NumpyBackendError> {
        let expected = rows
            .checked_mul(cols)
            .ok_or(NumpyBackendError::ShapeOverflow { rows, cols })?;
        if values.len() != expected {
            return Err(NumpyBackendError::LengthMismatch {
                label: "matrix",
                expected,
                actual: values.len(),
            });
        }
        Ok(Self { rows, cols, values })
    }

    /// Row count.
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Column count.
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// The row-major values, bit-exact.
    pub fn into_values(self) -> Vec<f32> {
        self.values
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NumpyBackendError {
    ShapeOverflow {
        rows: usize,
        cols: usize,
    },
    LengthMismatch {
        label: &'static str,
        expected: usize,
        actual: usize,
    },
    InnerDimensionMismatch {
        a_cols: usize,
        b_cols: usize,
    },
    Python(String),
}

impl fmt::Display for NumpyBackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeOverflow { rows, cols } => {
                write!(f, "matrix shape [{rows}, {cols}] overflows usize")
            }
            Self::LengthMismatch {
                label,
                expected,
                actual,
            } => write!(
                f,
                "{label} payload has {actual} values, expected {expected}"
            ),
            Self::InnerDimensionMismatch { a_cols, b_cols } => write!(
                f,
                "A @ B^T requires equal inner dimensions, got {a_cols} and {b_cols}"
            ),
            Self::Python(message) => write!(f, "NumPy operation failed: {message}"),
        }
    }
}

impl Error for NumpyBackendError {}

/// Eager `A @ B^T`, returning only Rust-owned values after releasing Python.
pub fn matmul_f32_abt(a: F32Matrix, b: F32Matrix) -> Result<F32Matrix, NumpyBackendError> {
    if a.cols != b.cols {
        return Err(NumpyBackendError::InnerDimensionMismatch {
            a_cols: a.cols,
            b_cols: b.cols,
        });
    }
    let rows = a.rows;
    let cols = b.rows;
    let output_len = rows
        .checked_mul(cols)
        .ok_or(NumpyBackendError::ShapeOverflow { rows, cols })?;
    let output_values = with_numpy(|py, numpy| {
        let a = numpy_f32_matrix(py, numpy, &a)?;
        let b = numpy_f32_matrix(py, numpy, &b)?;
        let b_transpose = b.getattr("T")?;
        let result = numpy.call_method1("matmul", (&a, &b_transpose))?;
        owned_rust_f32_values(py, numpy, &result, output_len)
    })?;
    F32Matrix::new(rows, cols, output_values)
}

fn with_numpy<T>(
    operation: impl for<'py> FnOnce(Python<'py>, &Bound<'py, PyModule>) -> PyResult<T>,
) -> Result<T, NumpyBackendError> {
    // Convert PyErr to a Rust-owned string before detaching from Python.
    // Thus even the error path retains no Python reference.
    Python::attach(|py| {
        let numpy = py.import("numpy").map_err(|error| error.to_string())?;
        operation(py, &numpy).map_err(|error| error.to_string())
    })
    .map_err(NumpyBackendError::Python)
}

fn numpy_f32_matrix<'py>(
    py: Python<'py>,
    numpy: &Bound<'py, PyModule>,
    matrix: &F32Matrix,
) -> PyResult<Bound<'py, PyAny>> {
    let bytes = f32_values_to_le_bytes(&matrix.values);
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", "<f4")?;
    let flat = numpy.call_method("frombuffer", (PyBytes::new(py, &bytes),), Some(&kwargs))?;
    flat.call_method1("reshape", ((matrix.rows, matrix.cols),))
}

fn owned_rust_f32_values(
    py: Python<'_>,
    numpy: &Bound<'_, PyModule>,
    value: &Bound<'_, PyAny>,
    expected: usize,
) -> PyResult<Vec<f32>> {
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", "<f4")?;
    let contiguous = numpy.call_method("ascontiguousarray", (value,), Some(&kwargs))?;
    let buffer = PyBuffer::<f32>::get(&contiguous)?;
    if buffer.item_count() != expected {
        return Err(PyValueError::new_err(format!(
            "NumPy returned {} values, expected {expected}",
            buffer.item_count()
        )));
    }
    buffer.to_vec(py)
}

fn f32_values_to_le_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// `MatmulBackend` over `numpy.matmul` (legacy `python`-feature path).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NumpyBackend;

impl MatmulBackend for NumpyBackend {
    fn matmul_f32_abt(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
    ) -> OpResult<()> {
        let to_op = |error: NumpyBackendError| OpError::message(error.to_string());
        let a = F32Matrix::new(m, k, a.to_vec()).map_err(to_op)?;
        let b = F32Matrix::new(n, k, b.to_vec()).map_err(to_op)?;
        let output = matmul_f32_abt(a, b).map_err(to_op)?.into_values();
        if c.len() != output.len() {
            return Err(OpError::message(format!(
                "matmul output has {} values, expected {}",
                c.len(),
                output.len()
            )));
        }
        c.copy_from_slice(&output);
        Ok(())
    }
}

// Needs an embedded interpreter: run with `--features numpy-embed`.
#[cfg(all(test, feature = "numpy-embed"))]
mod python_tests {
    use super::*;

    fn matrix(rows: usize, cols: usize, values: &[f32]) -> F32Matrix {
        F32Matrix::new(rows, cols, values.to_vec()).unwrap()
    }

    #[test]
    fn matmul_uses_b_transpose_and_returns_rust_owned_values() {
        let a = matrix(2, 3, &[1.0, 2.0, 3.0, -1.0, 0.5, 4.0]);
        let b = matrix(
            4,
            3,
            &[2.0, 0.0, 1.0, 1.0, 1.0, 1.0, -2.0, 3.0, 0.5, 0.0, -1.0, 2.0],
        );
        let expected = vec![5.0, 6.0, 5.5, 4.0, 2.0, 3.5, 5.5, 7.5];

        let output = matmul_f32_abt(a, b).unwrap();
        assert_eq!((output.rows(), output.cols()), (2, 4));
        assert_eq!(output.into_values(), expected);
    }

    #[test]
    fn matrices_and_results_hold_no_python_lifetime() {
        fn assert_send_sync_static<T: Send + Sync + 'static>() {}
        fn assert_send_static_future<T: std::future::Future + Send + 'static>(_: T) {}
        assert_send_sync_static::<F32Matrix>();

        assert_send_static_future(async move {
            let input = matrix(1, 3, &[1.0, 2.0, 4.0]);
            let output = matmul_f32_abt(input, matrix(1, 3, &[1.0; 3])).unwrap();
            std::future::ready(()).await;
            output
        });
        let output = std::thread::spawn(|| {
            let input = matrix(1, 3, &[1.0, 2.0, 4.0]);
            matmul_f32_abt(input, matrix(1, 3, &[1.0; 3])).unwrap()
        })
        .join()
        .unwrap();
        assert_eq!(output.into_values(), vec![7.0]);
    }

    #[test]
    fn shape_contracts_fail_before_numpy_execution() {
        assert_eq!(
            F32Matrix::new(2, 3, vec![0.0; 5]).unwrap_err(),
            NumpyBackendError::LengthMismatch {
                label: "matrix",
                expected: 6,
                actual: 5,
            }
        );
        let a = matrix(1, 2, &[1.0, 2.0]);
        let b = matrix(1, 3, &[1.0, 2.0, 3.0]);
        assert_eq!(
            matmul_f32_abt(a, b).unwrap_err(),
            NumpyBackendError::InnerDimensionMismatch {
                a_cols: 2,
                b_cols: 3,
            }
        );
    }

    #[test]
    fn numpy_backend_agrees_with_reference_on_exact_inputs() {
        use super::super::backend::ReferenceBackend;
        let (m, n, k) = (4, 5, 6);
        let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 - 3.0).collect();
        let b: Vec<f32> = (0..n * k).map(|i| (i % 5) as f32 - 2.0).collect();
        let mut numpy = vec![0.0; m * n];
        let mut reference = vec![0.0; m * n];
        NumpyBackend
            .matmul_f32_abt(m, n, k, &a, &b, &mut numpy)
            .unwrap();
        ReferenceBackend
            .matmul_f32_abt(m, n, k, &a, &b, &mut reference)
            .unwrap();
        assert_eq!(numpy, reference);
    }
}
