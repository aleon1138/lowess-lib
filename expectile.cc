#include <array>
#include <vector>
#include <cmath>
#include <immintrin.h>

#include "inc/nelder_mead.h"
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
    for (int i = 0; i < n; ++i) {
        u_buf[i] = (x[i] - x0) * k;
        w_buf[i] = expf(-0.5f * u_buf[i] * u_buf[i]);
    }

    float tol = 1e-6;
    int maxiter = 400;
    LossFunction loss(y, tau, n, u_buf.data(), w_buf.data());
    auto out = nelder_mead<float,2>(loss, {0,0}, tol*tol, {1,1}, 1, maxiter);
    return out.xmin[0];
}
