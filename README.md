# lowess-lib
Highly optimized LOWESS utilities for Python

This is a minimal python library for using local regression, also known as 
Locally Weighted Scatterplot Smoothing ([LOWESS](https://en.wikipedia.org/wiki/Local_regression)).

In my day-to-day work I use this extensively and just out of sheer repetition I 
need something that is as fast as possible, so I'm making use of [OpenMP](www.openmp.org) 
and [AVX](https://en.wikipedia.org/wiki/AVX-512) instructions. 

## Requirements

* [pybind11](https://github.com/pybind/pybind11) - for the C++ to python interface

## Installation

```
pip install .
```

## Performance

We can compare performance with the `smoothers_lowess` module from 
[statsmodels](https://www.statsmodels.org/dev/generated/statsmodels.nonparametric.smoothers_lowess.lowess.html#).

```python
import numpy as np
import statsmodels.api as sm
import lowesslib

x = np.random.uniform(low = -2*np.pi, high = 2*np.pi, size=10_000)
y = np.sin(x) + np.random.normal(size=len(x))
xi = np.linspace(-2*np.pi, 2*np.pi, 100)

%timeit z0 = sm.nonparametric.lowess(endog=y, exog=x, xvals=xi)
%timeit z1 = lowesslib.smooth(xi, x, y, 2.7)
```

Results are quite dramatic: over 20-thousand times faster than statsmodels.

```
statsmodels:
2.13 s ± 7.46 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

lowesslib:
99.5 µs ± 1.37 µs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
```
