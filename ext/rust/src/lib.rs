use numpy::ndarray::{azip, s, Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

///
///  Sorting can be a major bottleneck for very large arrays. So the simplest
///  approach is to just sub-sample the data and sort the reduced set.
///
pub fn subsample_sort(x: ArrayView1<f32>) -> Array1<f32> {
    const MAX_SIZE: usize = 100_000;
    let n = x.len();
    let stride = std::cmp::max(1, n / MAX_SIZE) as isize;

    let mut y = x.slice(s![..; stride]).to_owned();

    // Fail fast on NaNs (partial_cmp returns None → unwrap panics)
    y.as_slice_mut()
        .unwrap()
        .sort_by(|a, b| a.partial_cmp(b).unwrap());

    y
}

pub fn interquartile_range(x: ArrayView1<f32>) -> f32 {
    let x = subsample_sort(x);
    let n = x.len();
    x[(3 * n) / 4] - x[n / 4]
}

pub fn generate_bins(x: ArrayView1<f32>, bins: usize) -> Array1<f32> {
    let x = subsample_sort(x);
    let bins = x.len().min(bins);
    let slope = (x.len().saturating_sub(1) as f32) / ((bins + 1) as f32);

    let mut out = Array1::<f32>::zeros(bins);
    for i in 0..bins {
        let k = (((i + 1) as f32) * slope).round() as usize;
        out[i] = x[k];
    }
    out
}

///
///  Weighted local-linear intercept at x0 with Gaussian kernel of scale h.
///
pub fn solve_intercept(x: ArrayView1<f32>, y: ArrayView1<f32>, x0: f32, h: f32) -> f32 {
    assert_eq!(x.len(), y.len(), "x and y must have same length");
    assert!(h > 0.0, "h must be > 0");

    let k = 1.0_f32 / h;

    // Accumulators (coeff_t)
    let (mut x00, mut x01, mut x11, mut xy0, mut xy1) = (0.0f32, 0.0, 0.0, 0.0, 0.0);

    azip!((&xi in x, &yi in y) {
        let u  = (x0 - xi) * k;
        let u2 = u * u;
        let w2 = (-u2).exp();
        x00 += w2;
        x01 += w2 * u;
        x11 += w2 * u2;
        xy0 += w2 * yi;
        xy1 += w2 * yi * u;
    });

    let numer = x11 * xy0 - x01 * xy1;
    let denom = x00 * x11 - x01 * x01;

    if denom > 0.0 {
        numer / denom
    } else {
        0.0
    }
}

///
///  Perform LOWESS (Locally Weighted Scatterplot Smoothing) on one-dimensional data.
///
///  Parameters
///  ----------
///  x : array_like, shape (N,)
///      Independent variable.
///
///  y : array_like, shape (N,)
///      Dependent (response) variable.
///
///  bins : int or sequence of scalars, optional
///      If an integer, specifies the number of quantiles in `x` to use as
///      interpolation points. If a sequence, it is treated as the exact
///      interpolation point locations.
///
///  bandwidth : float, optional
///      Kernel bandwidth for smoothing, in the same units as `x`.
///      Defaults to 1.41 times the interquartile range (IQR) of `x`.
///
///  Returns
///  -------
///  xi : ndarray
///      Interpolation point locations in `x`.
///
///  yi : ndarray
///      Smoothed, interpolated values of `y` at `xi`.)
///
#[pyfunction]
#[pyo3(signature = (x, y, bins=100, bandwidth=None))]
fn smooth<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f32>,
    y: PyReadonlyArray1<'py, f32>,
    bins: usize,
    bandwidth: Option<f32>,
) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>)> {
    let x = x.as_array();
    let y = y.as_array();
    if x.len() != y.len() {
        return Err(PyValueError::new_err("input length mismatch"));
    }

    let x_out = generate_bins(x, bins);
    let h = bandwidth.unwrap_or_else(|| interquartile_range(x) * 1.414);
    if h <= 0.0 {
        return Err(PyValueError::new_err("invalid bandwidth"));
    }

    let mut y_out = Vec::with_capacity(x_out.len());
    for &xi in x_out.iter() {
        y_out.push(solve_intercept(x, y, xi, h));
    }

    Ok((x_out.into_pyarray(py), PyArray1::from_vec(py, y_out)))
}

/// LOWESS: Locally Weighted Scatter plot Smoothing
#[pymodule]
fn lowesslib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(smooth, m)?)?;
    Ok(())
}
