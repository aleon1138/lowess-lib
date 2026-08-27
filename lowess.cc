#include <cmath>


/*
 * Accumulated in double. The intercept is recovered from
 * `(x11*xy0 - x01*xy1) / (x00*x11 - x01*x01)`, and that denominator suffers
 * heavy cancellation whenever the local window sits off to one side of the
 * data (`x01*x01` approaches `x00*x11`). Summing hundreds of thousands of
 * terms in float leaves fewer significant digits than the cancellation eats,
 * so the tails of the fit degenerate into noise.
 */
struct coeff_t {
    double x00;
    double x01;
    double x11;
    double xy0;
    double xy1;
};


struct covar_t {
    float xx;
    float xy;
};


#ifdef __AVX__
#include <immintrin.h>
#include <xmmintrin.h>


/*
 *  Gaussian kernel:
 *    y = exp(-0.5*u*u)
 */
__m256 _mm256_gauss_kernel_ps(__m256 u)
{
    __m256 x = _mm256_mul_ps(_mm256_set1_ps(-0.5f), _mm256_mul_ps(u,u));

    // Clamp `x` to avoid numerical issues with large negative values
    x = _mm256_max_ps(x, _mm256_set1_ps(-30.0f));

    // Fast approximation for exp(x)
    // See: stackoverflow.com/q/47025373
    __m256  a = _mm256_set1_ps(12102203.1615614f); // (1 << 23) / log(2)
    __m256i b = _mm256_set1_epi32((127 << 23) - 298765);
    __m256i t = _mm256_add_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(a, x)), b);
    return _mm256_castsi256_ps(t);
}


/*
 *  Horizontal sum
 *  See: stackoverflow.com/q/6996764
 */
double hsum(__m256d v)
{
    __m128d lo   = _mm256_castpd256_pd128(v);
    __m128d hi   = _mm256_extractf128_pd(v, 1);
    __m128d lohi = _mm_add_pd(lo, hi);
    return _mm_cvtsd_f64(_mm_add_sd(lohi, _mm_unpackhi_pd(lohi, lohi)));
}


float hsum(__m256 v)
{
    __m128 lo   = _mm256_castps256_ps128(v);
    __m128 hi   = _mm256_extractf128_ps(v, 1);
    __m128 lohi = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(lohi);
    __m128 sums = _mm_add_ps(lohi, shuf);
    return _mm_cvtss_f32(_mm_add_ss(sums, _mm_movehl_ps(shuf, sums)));
}


/*
 *  Accumulate the weighted normal-equation sums for local linear regression using SIMD
 */
coeff_t solve_intercept_simd(const float *x_, const float *y_, float x0_, float h, int n)
{
    __m256 x0 = _mm256_set1_ps(x0_);
    __m256 k  = _mm256_set1_ps(1.0f / h);

    __m256d x00 = _mm256_setzero_pd();
    __m256d x01 = _mm256_setzero_pd();
    __m256d x11 = _mm256_setzero_pd();
    __m256d xy0 = _mm256_setzero_pd();
    __m256d xy1 = _mm256_setzero_pd();

    for (int i = 0; i < n; i += 8) {
        __m256 x   = _mm256_loadu_ps(x_+i);
        __m256 yf  = _mm256_loadu_ps(y_+i);
        __m256 u   = _mm256_mul_ps(_mm256_sub_ps(x0, x), k);
        __m256 w   = _mm256_gauss_kernel_ps(u);
        __m256 w2f = _mm256_mul_ps(w, w);

        // Widen once, then build the five sums with FMA in double
        for (int half = 0; half < 2; ++half) {
            __m128 w2s = half? _mm256_extractf128_ps(w2f,1) : _mm256_castps256_ps128(w2f);
            __m128 us  = half? _mm256_extractf128_ps(u,  1) : _mm256_castps256_ps128(u);
            __m128 ys  = half? _mm256_extractf128_ps(yf, 1) : _mm256_castps256_ps128(yf);
            __m256d w2 = _mm256_cvtps_pd(w2s);
            __m256d ud = _mm256_cvtps_pd(us);
            __m256d yd = _mm256_cvtps_pd(ys);
            __m256d w2u = _mm256_mul_pd(w2, ud);
            x00 = _mm256_add_pd (x00, w2);
            x01 = _mm256_add_pd (x01, w2u);
            x11 = _mm256_fmadd_pd(w2u, ud, x11);
            xy0 = _mm256_fmadd_pd(w2,  yd, xy0);
            xy1 = _mm256_fmadd_pd(w2u, yd, xy1);
        }
    }

    // *INDENT-OFF*
    return coeff_t {
        x00: hsum(x00),
        x01: hsum(x01),
        x11: hsum(x11),
        xy0: hsum(xy0),
        xy1: hsum(xy1),
    };
    // *INDENT-ON*
}


float histogram_kernel_simd(const float *x_, float x0_, float h, int n)
{
    __m256 s  = _mm256_setzero_ps();
    __m256 k  = _mm256_set1_ps(1.0f / h);
    __m256 x0 = _mm256_set1_ps(x0_);

    for (int i = 0; i < n; i += 8) {
        __m256 x = _mm256_loadu_ps(x_+i);
        __m256 u = _mm256_mul_ps(_mm256_sub_ps(x0, x), k);
        s = _mm256_add_ps(_mm256_gauss_kernel_ps(u), s);
    }
    return hsum(s);
}


covar_t interact_kernel_simd(const float *x_, const float *y_, const float *z_,
                             float z0_, float h, int n)
{
    __m256 xx = _mm256_setzero_ps();
    __m256 xy = _mm256_setzero_ps();
    __m256 k  = _mm256_set1_ps(1.0f / h);
    __m256 z0 = _mm256_set1_ps(z0_);

    for (int i = 0; i < n; i += 8) {
        __m256 z = _mm256_loadu_ps(z_+i);
        __m256 u = _mm256_mul_ps(_mm256_sub_ps(z0, z), k);
        __m256 w = _mm256_gauss_kernel_ps(u);
        __m256 xw = _mm256_mul_ps(_mm256_loadu_ps(x_+i), w);
        __m256 yw = _mm256_mul_ps(_mm256_loadu_ps(y_+i), w);
        xx = _mm256_fmadd_ps(xw, xw, xx);
        xy = _mm256_fmadd_ps(xw, yw, xy);
    }

    // *INDENT-OFF*
    return covar_t {
        xx : hsum(xx),
        xy : hsum(xy),
    };
    // *INDENT-ON*
}
#endif


float solve_intercept(const float *x, const float *y, float x0, float h, int n)
{
#ifdef __AVX__
    int n0 = n - (n%8);
    coeff_t o = solve_intercept_simd(x, y, x0, h, n0);
#else
    int n0 = 0;
    coeff_t o = {0};
#endif

    float k = 1.0f / h;
    for (int i = n0; i < n; ++i) {
        double u  = (x0 - x[i]) * k;
        double w  = exp(-0.5 * u * u);
        double w2 = w * w;
        o.x00 += w2;
        o.x01 += w2 * u;
        o.x11 += w2 * u * u;
        o.xy0 += w2 * y[i];
        o.xy1 += w2 * y[i] * u;
    }

    double numer = o.x11 * o.xy0 - o.x01 * o.xy1;
    double denom = o.x00 * o.x11 - o.x01 * o.x01;

    /*
     * `denom` is a difference of two nearly equal products, so testing it
     * against 0 is not enough: once the local window carries no real spread in
     * `u`, what survives the subtraction is pure round-off of either sign, and
     * dividing by it yields noise. Compare against the scale of the terms
     * instead. Windows that are merely off to one side of the data still
     * retain ~1e-5 of that scale, while genuinely degenerate ones land at
     * ~1e-16, so the cutoff sits well clear of both.
     *
     * Report degeneracy as NaN rather than 0 — 0 is a perfectly plausible
     * estimate and silently blends in with the rest of the curve.
     */
    const double COND_TOL = 1e-12;
    return denom > COND_TOL * o.x00 * o.x11? numer / denom : NAN;
}


float histogram_kernel(const float *x, float x0, float h, int n)
{
#ifdef __AVX__
    int n0 = n - (n%8);
    float s = histogram_kernel_simd(x, x0, h, n0);
#else
    int n0 = 0;
    float s = 0.0f;
#endif

    float k = 1.0f / h;
    for (int i = n0; i < n; ++i) {
        float u = (x0 - x[i]) * k;
        s += expf(-0.5f * u * u);
    }

    const float K0 = 0.3989422804014327f; // 1/sqrt(2*pi)
    return K0 * s / (n * h);
}


float interact_kernel(const float *x, const float *y, const float *z,
                      float z0, float h, int n)
{
#ifdef __AVX__
    int n0 = n - (n%8);
    covar_t o = interact_kernel_simd(x, y, z, z0, h, n0);
#else
    int n0 = 0;
    covar_t o = {0};
#endif

    for (int i = n0; i < n; ++i) {
        float u = (z0 - z[i]) / h;
        float w = expf(-0.5 * u * u);
        o.xx += x[i] * x[i] * w * w;
        o.xy += x[i] * y[i] * w * w;
    }
    return o.xx > 0.0? o.xy / o.xx : NAN;  // see solve_intercept()
}
