#include <vector>
#include <cmath>

#include "inc/kernel.h"

#ifdef __AVX__
#include <immintrin.h>
double hsum(__m256d v);  // lowess.cc
#endif

/*
 * How many IRLS passes to allow. The expectile loss is convex and piecewise
 * quadratic, so the iteration is a fixed point on the sign pattern of the
 * residuals and settles in a handful of passes; this is a backstop, not an
 * expected exit.
 */
#define MAX_IRLS_ITER 50


/*
 * Weighted normal-equation sums for a local linear fit, accumulated in double
 * for the reason given in lowess.cc: the solve divides by
 * `s00*s11 - s01*s01`, which cancels heavily once the window sits off to one
 * side of the data.
 */
struct wsum_t {
    double s00;
    double s01;
    double s11;
    double sy0;
    double sy1;
    double syy;
};


/*
 * Accumulate one IRLS pass. The expectile weight `t` is `tau` where the
 * residual under the current fit `(a, b)` is non-negative and `1-tau`
 * elsewhere, so the sums depend on the fit and have to be rebuilt each pass.
 *
 * At `tau = 0.5` the weight is constant and this reduces to ordinary weighted
 * least squares, independent of `a` and `b`.
 */
static wsum_t expectile_sums(const float *y_, const float *u_, const float *w_,
                             int n, float tau, float a, float b)
{
    wsum_t o = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    int i = 0;

#ifdef __AVX__
    __m256d v00 = _mm256_setzero_pd();
    __m256d v01 = _mm256_setzero_pd();
    __m256d v11 = _mm256_setzero_pd();
    __m256d vy0 = _mm256_setzero_pd();
    __m256d vy1 = _mm256_setzero_pd();
    __m256d vyy = _mm256_setzero_pd();

    const __m256 va     = _mm256_set1_ps(a);
    const __m256 vb     = _mm256_set1_ps(b);
    const __m256 vtau   = _mm256_set1_ps(tau);
    const __m256 v1mtau = _mm256_set1_ps(1.0f - tau);
    const __m256 vzero  = _mm256_setzero_ps();

    for (; i <= n - 8; i += 8) {
        __m256 u = _mm256_loadu_ps(u_+i);
        __m256 y = _mm256_loadu_ps(y_+i);
        __m256 w = _mm256_loadu_ps(w_+i);

        __m256 e    = _mm256_sub_ps(y, _mm256_fmadd_ps(vb, u, va));
        __m256 mask = _mm256_cmp_ps(e, vzero, _CMP_GE_OQ);
        __m256 t    = _mm256_blendv_ps(v1mtau, vtau, mask);
        __m256 wt   = _mm256_mul_ps(t, _mm256_mul_ps(w, w));

        // Widen once, then build the sums with FMA in double
        for (int half = 0; half < 2; ++half) {
            __m128 ws = half? _mm256_extractf128_ps(wt, 1) : _mm256_castps256_ps128(wt);
            __m128 us = half? _mm256_extractf128_ps(u,  1) : _mm256_castps256_ps128(u);
            __m128 ys = half? _mm256_extractf128_ps(y,  1) : _mm256_castps256_ps128(y);
            __m256d wd = _mm256_cvtps_pd(ws);
            __m256d ud = _mm256_cvtps_pd(us);
            __m256d yd = _mm256_cvtps_pd(ys);
            __m256d wu = _mm256_mul_pd(wd, ud);
            v00 = _mm256_add_pd  (v00, wd);
            v01 = _mm256_add_pd  (v01, wu);
            v11 = _mm256_fmadd_pd(wu, ud, v11);
            vy0 = _mm256_fmadd_pd(wd, yd, vy0);
            vy1 = _mm256_fmadd_pd(wu, yd, vy1);
            vyy = _mm256_fmadd_pd(_mm256_mul_pd(wd, yd), yd, vyy);
        }
    }

    o.s00 = hsum(v00);
    o.s01 = hsum(v01);
    o.s11 = hsum(v11);
    o.sy0 = hsum(vy0);
    o.sy1 = hsum(vy1);
    o.syy = hsum(vyy);
#endif

    for (; i < n; ++i) {
        double u = u_[i];
        double y = y_[i];
        double w = w_[i];
        double e = y - (a + b * u);
        double t = e >= 0.0? tau : 1.0 - tau;
        double wt = t * w * w;
        o.s00 += wt;
        o.s01 += wt * u;
        o.s11 += wt * u * u;
        o.sy0 += wt * y;
        o.sy1 += wt * u * y;
        o.syy += wt * y * y;
    }
    return o;
}


/*
 * Solve the 2x2 normal equations for the intercept and slope. Returns false
 * when the system is too ill-conditioned to trust.
 */
static bool solve_normal_equations(const wsum_t &o, double *a, double *b)
{
    double denom = o.s00 * o.s11 - o.s01 * o.s01;
    if (!(denom > COND_TOL * o.s00 * o.s11)) {
        return false;
    }
    *a = (o.s11 * o.sy0 - o.s01 * o.sy1) / denom;
    *b = (o.s00 * o.sy1 - o.s01 * o.sy0) / denom;
    return true;
}


float solve_expectile(const float *x, const float *y, float x0,
                      float h, float tau, int n)
{
    thread_local std::vector<float> u_buf, w_buf;
    if ((int)u_buf.size() < n) {
        u_buf.resize(n);
        w_buf.resize(n);
    }

    const float k = 1.0f / h;
    double sw2 = 0.0, sw2u = 0.0;
    for (int i = 0; i < n; ++i) {
        double u = (x[i] - x0) * k;
        double w = gauss_kernel(u);
        u_buf[i] = u;
        w_buf[i] = w;
        sw2  += w * w;
        sw2u += w * w * u;
    }

    /*
     * Same reasoning as solve_intercept(): this reports the fit at `u = 0`, so
     * any error in the slope is amplified by how far the supporting data sits
     * from the point being estimated. `sw2u/sw2` measures that distance in
     * bandwidths. The kernel weights do not involve `tau`, so which bins are
     * refused is a property of `x` alone and is identical at every `tau`.
     */
    if (!(sw2 > 0) || fabs(sw2u / sw2) > MAX_EXTRAPOLATION) {
        return NAN;
    }

    /*
     * Seed with ordinary weighted least squares, which is the `tau = 0.5`
     * case. This is the same system solve_intercept() solves, so
     * `expectile(tau=0.5)` and `smooth()` agree to ~3e-4 -- they differ only
     * because solve_intercept() weights with the fast SIMD exp approximation
     * while the kernel weights here come from exp() directly.
     */
    double a, b;
    wsum_t o = expectile_sums(y, u_buf.data(), w_buf.data(), n, 0.5f, 0.0f, 0.0f);
    if (!solve_normal_equations(o, &a, &b)) {
        return NAN;
    }
    if (tau == 0.5f) {
        return a;
    }

    /*
     * Iteratively reweighted least squares. Each pass fixes the expectile
     * weights from the current residuals and solves the resulting weighted
     * least squares problem. The loss is convex and piecewise quadratic, so
     * once the sign pattern of the residuals stops changing the fit is exact.
     *
     * The convergence test is relative to the weighted RMS of `y` in the
     * window, not a fixed number. An absolute tolerance is really a tolerance
     * on the scale of `y` and on how much kernel weight the window happens to
     * carry: that is what made the previous Nelder-Mead solver quit early on
     * sparse windows and return a fit that was not scale-equivariant.
     */
    const double scale = sqrt(o.syy / o.s00);
    const double TOL = 1e-6;

    for (int iter = 0; iter < MAX_IRLS_ITER; ++iter) {
        double a_prev = a, b_prev = b;
        o = expectile_sums(y, u_buf.data(), w_buf.data(), n, tau, a, b);
        if (!solve_normal_equations(o, &a, &b)) {
            return NAN;
        }
        if (fabs(a - a_prev) + fabs(b - b_prev) <= TOL * scale) {
            break;
        }
    }
    return a;
}
