#include <array>
#include <vector>
#include <cmath>

//#include <cstdio>

#include "ext/nelder_mead/nelder_mead.h"

struct LossFunction {
    const float *_y;
    const float _tau;
    const int _n;
    std::vector<float> _w;
    std::vector<float> _u;

    LossFunction(const float *x, const float *y, float x0, float h, float tau, int n)
        : _y(y), _tau(tau), _n(n), _w(n), _u(n)
    {
        for (int i = 0; i < _n; ++i) {
            _u[i] = (x[i] - x0) / h;
            _w[i] = expf(-0.5f * _u[i] * _u[i]);
        }
    }

    double operator () (const std::array<double,2> &theta) const
    {
        float theta_0 = theta[0];
        float theta_1 = theta[1];
        double loss = 0.0;
        for (int i = 0; i < _n; ++i) {
            float e = _y[i] - (theta_0 + theta_1 * _u[i]);
            float t = e >= 0.0f? _tau : 1.0f - _tau;
            loss += t * (e * _w[i]) * (e * _w[i]);
        }
        return loss / _n;
    }
};


float solve_expectile(const float *x, const float *y, float x0,
                      float h, float tau, int n)
{
    // * reqmin should remain a tiny number like 1e-18
    // * simplex size should be interquartile range of x and y (or is it theta??)
    //   if the symplex size is indeed is in units of theta perhaps we need to
    //   run a regression first to get the units of theta
    // * theta should be "double" type
    // * it does not look **too** sensitive to simplex size?

    LossFunction loss(x, y, x0, h, tau, n);
    auto out = nelder_mead<double,2>(loss, {0,0}, 1e-18, {1,1});

    //fprintf(stderr, "%d %d\n", out.icount, out.numres);
    //fflush(stderr);

    return out.xmin[0];
}
