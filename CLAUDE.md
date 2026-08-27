# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## Project Overview

**lowesslib** is a high-performance Python library for LOWESS (Locally Weighted
Scatterplot Smoothing) implemented in C++17 with AVX2/FMA SIMD and OpenMP
parallelization. It exposes `smooth()`, `histogram()`, `interact()`, and
`expectile()` to Python via pybind11.

## Build

Requires: C++17 compiler with AVX2/OpenMP support, pybind11.

```bash
git clone https://github.com/aleon1138/lowess-lib.git

# Install (builds the C++ extension in-place)
pip install .

# Or build directly via Make
make
```

## Testing

```bash
# Build the extension and run both the Python and C++ suites
make test

# Python tests only (requires lowesslib.so in the repo root, i.e. `make`)
python -m pytest tests/ -v

# Run a single test
python -m pytest tests/test_lowess.py::TestLowess::test_smooth_avx -v
```

Tests compare the C++ extension against a Numba reference implementation
(`ext/lowesslib_numba.py`) and SciPy for expectile regression.

`tests/test_lowess.py` also defines a `benchmark()` helper, which is not
collected by pytest — import and call it directly to time `smooth()` against
the Numba reference across a range of input sizes.

## C++ Formatting

```bash
make format   # runs astyle with project flags (-A4 -S -z2 -n -j)
```

## Architecture

The library has three C++ source files:

- **`lowess.cc`** — Core SIMD kernel. Contains the AVX2 Gaussian kernel
  (`_mm256_gauss_kernel_ps`), fast exp approximation, and
  `solve_intercept_simd()` (weighted least squares). Has scalar fallback for
  non-AVX systems.

  The weights are computed in float but the normal-equation sums **must** be
  accumulated in double. `solve_intercept()` divides by
  `x00*x11 - x01*x01`, which nearly cancels whenever the local window sits off
  to one side of the data; float accumulation over a large `n` does not leave
  enough significant digits to survive it, and the tails of the fit degenerate
  into noise.

  Three guards decide when a fit is refused (returning NaN, never 0):
  `GAUSS_CUTOFF` truncates the kernel to compact support, so a window with no
  nearby data is genuinely empty rather than sharing one floor weight;
  `MAX_EXTRAPOLATION` bounds `|x01/x00|`, the distance in bandwidths to the
  supporting data's centre of mass, which is also the factor amplifying any
  slope error; and `COND_TOL` bounds the cancellation in the denominator.
  `ext/lowesslib_numba.py` mirrors all three and the tests compare against it,
  so the constants must be kept in sync.
- **`expectile.cc`** — Expectile regression by IRLS. The loss is convex and
  piecewise quadratic, so `expectile_sums()` fixes the asymmetric weights from
  the current residuals and `solve_normal_equations()` solves the resulting
  weighted least squares problem; it converges in ~5 passes. `tau = 0.5` makes
  the weights constant, so it is the same system `solve_intercept()` solves and
  is returned directly from the seed pass.

  The convergence test **must** stay relative — it is measured against the
  weighted RMS of `y` in the window. An absolute tolerance is really a
  tolerance on the scale of `y` and on how much kernel weight the window
  carries, which is what made the previous Nelder-Mead solver quit early on
  sparse windows and return fits that were not scale-equivariant.

- **`inc/kernel.h`** — `GAUSS_CUTOFF`, `MAX_EXTRAPOLATION`, `COND_TOL` and
  `gauss_kernel()`, shared by `lowess.cc` and `expectile.cc`. Every estimator
  has to agree on where the kernel stops and on when a fit is refused.
- **`lowesslib.cc`** — Pybind11 bindings. All public API lives here: input
  validation, NaN handling, bin generation, and `parallel_apply()` which
  dispatches kernel calls across OpenMP threads.

The key performance techniques:
- AVX2 processes 8 floats in parallel per kernel evaluation
- OpenMP parallelizes across the output interpolation points
- Arrays larger than `MAX_SIZE` (100,000) are sub-sampled before sorting to
  avoid the O(n log n) bottleneck

