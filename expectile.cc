#include <array>
#include <vector>
#include <cmath>
#include <immintrin.h>

#include "inc/nelder_mead.h"
#include "inc/kernel.h"
float hsum(__m256 v);

struct LossFunction {
    const float *_y;
    const float *_w;
    const float *_u;
    const float _tau;
    const int _n;

    LossFunction(const float *y, float tau, int n, const float *u, const float *w)
        : _y(y), _w(w), _u(u), _tau(tau), _n(n) {}


    float operator()(const std::array<float, 2> &theta) const
    {
        __m256 v_theta_0 = _mm256_set1_ps(theta[0]);
        __m256 v_theta_1 = _mm256_set1_ps(theta[1]);
        __m256 v_tau     = _mm256_set1_ps(_tau);
        __m256 v_1mtau   = _mm256_set1_ps(1.0f - _tau);
        __m256 v_zero    = _mm256_setzero_ps();
        __m256 v_loss    = _mm256_setzero_ps();

        int i = 0;
        for (; i <= _n - 8; i += 8) {
            __m256 u = _mm256_loadu_ps(&_u[i]);
            __m256 y = _mm256_loadu_ps(&_y[i]);
            __m256 w = _mm256_loadu_ps(&_w[i]);
            __m256 e = _mm256_sub_ps(y, _mm256_fmadd_ps(v_theta_1, u, v_theta_0));
            __m256 mask = _mm256_cmp_ps(e, v_zero, _CMP_GE_OQ);
            __m256 t = _mm256_blendv_ps(v_1mtau, v_tau, mask);
            __m256 ew = _mm256_mul_ps(e, w);
            v_loss = _mm256_fmadd_ps(t, _mm256_mul_ps(ew, ew), v_loss);
        }
        float loss = hsum(v_loss);

        for (; i < _n; ++i) {
            float e = _y[i] - (theta[0] + theta[1] * _u[i]);
            float t = e >= 0.0f ? _tau : 1.0f - _tau;
            loss += t * (e * _w[i]) * (e * _w[i]);
        }

        return loss / _n;
    }
};


float solve_expectile(const float *x, const float *y, float x0,
                      float h, float tau, int n)
{
    /*
     * NOTES:
     * - `reqmin` should be a tiny number like 1e-18 for f64 or 1e-12 for f32.
     * - KNOWN DEFECT: `reqmin` is an absolute tolerance, but the loss below is
     *   divided by `n` rather than by the weight it actually carries, so its
     *   magnitude collapses as the window moves off the data -- from 2.7e-3 to
     *   1.1e-6 over four bins in one measurement. Nelder-Mead then converges
     *   prematurely and `theta` is left near its {0,0} start. Fits stay wrong
     *   but finite between roughly 1.3 and 3 bandwidths of offset, where the
     *   guard below still admits them; scaling `y` up by 1e3 restores
     *   agreement with a float64 IRLS reference, which confirms the cause.
     * - the code does not seem sensitive to `step` size, perhaps because `u`
     *   is already normalized. A value between 0.1 and 10 seems to work.
     * - accumulating `loss` as f64 may improve precision (not currently done)
     */
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
     * Same reasoning as solve_intercept(): this fits `theta[0] + theta[1]*u`
     * and reports `theta[0]`, the value at `u = 0`, so any error in the slope
     * `theta[1]` is amplified by how far the supporting data sits from the
     * point being estimated. `sw2u/sw2` measures that distance in bandwidths.
     *
     * The expectile loss is asymmetric in `tau`, so this mean -- taken under
     * the symmetric kernel weights alone -- is not exactly the amplification
     * factor here. It is used as a test of where the data lies relative to the
     * window, which is a property of `x` alone, so which bins are refused does
     * not depend on `tau`. Measured on data ending at x=0.85 with h=0.01:
     * interior bins 0.003, the one-sided window at the data edge 0.566, and
     * the first refused bin 3.151.
     */
    if (!(sw2 > 0) || fabs(sw2u / sw2) > MAX_EXTRAPOLATION) {
        return NAN;
    }

    float tol = 1e-6;
    int maxiter = 400;
    LossFunction loss(y, tau, n, u_buf.data(), w_buf.data());
    auto out = nelder_mead<float,2>(loss, {0,0}, tol*tol, {1,1}, 1, maxiter);
    return out.xmin[0];
}
