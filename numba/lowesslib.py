"""
LOWESS: Locally Weighted Scatter plot Smoothing
"""

import numpy as np
import numba


def _subsample_sort(x):
    step = max(1, len(x) // 100_000)
    return np.sort(x[::step])


def _interquartile_range(x):
    x = _subsample_sort(x)
    n = len(x)
    return x[n * 3 // 4] - x[n // 4]


def _process_bins_array(x, bins):
    if hasattr(bins, "__len__"):
        return np.array(bins, dtype="f32").squeeze()
    x = _subsample_sort(x)
    bins = min(bins, len(x))
    slope = float(len(x) - 1) / float(bins + 1)
    return np.array([x[round(float(i + 1) * slope)] for i in range(bins)])


@numba.njit(parallel=True)
def _solve_intercept(x, y, x_out, h):
    k = 1.0 / h
    y_out = np.zeros(len(x_out), dtype="f")
    for j in numba.prange(len(x_out)):
        x00, x01, x11, xy0, xy1 = 0, 0, 0, 0, 0
        for i in range(len(x)):
            u = (x_out[j] - x[i]) * k
            w = np.exp(-u * u)
            x00 += w
            x01 += w * u
            x11 += w * u * u
            xy0 += w * y[i]
            xy1 += w * y[i] * u

        numer = x11 * xy0 - x01 * xy1
        denom = x00 * x11 - x01 * x01
        if denom > 0:
            y_out[j] = numer / denom
    return y_out


def smooth(x, y, bins=100, bandwidth=None):
    """
    Perform LOWESS (Locally Weighted Scatterplot Smoothing) on one-dimensional data.

    Parameters
    ----------
    x : array_like, shape (N,)
        Independent variable.

    y : array_like, shape (N,)
        Dependent (response) variable.

    bins : int or sequence of scalars, optional
        If an integer, specifies the number of quantiles in `x` to use as
        interpolation points. If a sequence, it is treated as the exact
        interpolation point locations.

    bandwidth : float, optional
        Kernel bandwidth for smoothing, in the same units as `x`.
        Defaults to 1.41 times the interquartile range (IQR) of `x`.

    Returns
    -------
    xi : ndarray
        Interpolation point locations in `x`.

    yi : ndarray
        Smoothed, interpolated values of `y` at `xi`.
    """

    assert len(x) == len(y), "input length mismatch"

    x_out = _process_bins_array(x, bins)
    h = bandwidth if bandwidth is not None else _interquartile_range(x) * 1.414
    assert h > 0, "invalid bandwidth"
    return x_out, _solve_intercept(x, y, x_out, h)
