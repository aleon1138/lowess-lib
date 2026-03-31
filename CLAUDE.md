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
# Run all tests
python test.py

# The test file also contains benchmarks — run a specific test class/method:
python -m unittest test.TestLowess.test_smooth_avx
```

Tests compare the C++ extension against a Numba reference implementation
(`ext/lowesslib_numba.py`) and SciPy for expectile regression.

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
- **`expectile.cc`** — Expectile regression using the Nelder-Mead optimizer
  from `inc/nelder_mead.h`. `solve_expectile()` calls the `LossFunction` struct
  which uses AVX2 internally.
- **`lowesslib.cc`** — Pybind11 bindings. All public API lives here: input
  validation, NaN handling, bin generation, and `parallel_apply()` which
  dispatches kernel calls across OpenMP threads.

The key performance techniques:
- AVX2 processes 8 floats in parallel per kernel evaluation
- OpenMP parallelizes across the output interpolation points
- Arrays larger than `MAX_SIZE` (100,000) are sub-sampled before sorting to
  avoid the O(n log n) bottleneck

