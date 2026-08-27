#ifndef LOWESSLIB_KERNEL_H
#define LOWESSLIB_KERNEL_H

#include <cmath>

/*
 * Shared between `lowess.cc` and `expectile.cc`. Every estimator in the
 * library has to agree on where the kernel stops and on when a fit is too
 * poorly supported to report, so these live in one place rather than being
 * restated per translation unit.
 */

/*
 * The Gaussian is truncated at exp(-GAUSS_CUTOFF), i.e. it has compact support
 * over |u| < sqrt(2*GAUSS_CUTOFF), about 7.75 bandwidths. Past that the weight
 * is below 1e-13 and contributes nothing to a well-posed fit, but keeping it
 * non-zero is actively harmful -- see _mm256_gauss_kernel_ps().
 *
 * The SIMD and scalar paths must agree on this cutoff, or results will shift
 * depending on where an input happens to fall relative to a multiple of 8.
 */
#define GAUSS_CUTOFF 30.0f


/*
 * How far, in bandwidths, the data supporting a fit may sit from the point
 * being estimated before the fit is refused. See solve_intercept() for why
 * this is the natural quantity to bound. Interior windows sit at ~0 and a
 * one-sided window at a true data boundary at 1/sqrt(pi) ~= 0.56, so 3 leaves
 * legitimate edge fits untouched.
 */
#define MAX_EXTRAPOLATION 3.0


inline double gauss_kernel(double u)
{
    double uu = u * u;
    return uu < 2.0 * GAUSS_CUTOFF? exp(-0.5 * uu) : 0.0;
}

#endif
