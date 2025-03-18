#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <omp.h>

namespace py = pybind11;
typedef py::array_t<float, py::array::f_style | py::array::forcecast> array_t;

float solve_intercept(const float *x, const float *y, float x0, float h, int n);
float histogram_kernel(const float *x, float x0, float h, int n);

/*
 *  Approximate interquartile range via gradient descent. This idea is cool
 *  but it's not working very reliably. Still needs some work.
 */
float interquartile_range_approx(const array_t &x)
{
    const float *px  = x.data(0);
    int n = x.shape(0);

    float q25(0), q75(0);
    float w = 1.0f/float(n);
    for (int i = 0; i < n; ++i) {
        q25 += w * ((q25 > px[i])? -0.75f : 0.25f);
        q75 += w * ((q75 > px[i])? -0.25f : 0.75f);
    }
    return q75 - q25;
}

float interquartile_range(const array_t &x)
{
    const int n = x.shape(0);
    std::vector<float> x_sort(x.data(0), x.data(0) + n);
    std::sort(x_sort.begin(), x_sort.end());
    return x_sort[n*3/4] - x_sort[n/4];
}

void verify(bool cond, const char*msg)
{
    if (cond == false) {
        throw std::invalid_argument(msg);
    }
}

py::array_t<float> smooth(array_t xi, array_t x, array_t y, std::optional<float> bandwidth)
{
    verify(x.shape(0) == y.shape(0), "x and y are not of the same length");
    verify(x.shape(0) > 0, "x is empty");
    verify(xi.ndim() == 1, "xi must be 1-dimensional");
    verify(x.ndim()  == 1, "x must be 1-dimensional");
    verify(y.ndim()  == 1, "y must be 1-dimensional");

    py::array_t<float> yi = py::array_t<float>(xi.shape(0));
    float *pyi = yi.mutable_data(0);

    const float *pxi = xi.data(0);
    const float *px  = x.data(0);
    const float *py  = y.data(0);
    int m = xi.shape(0);
    int n = x.shape(0);

    float h;
    if (bandwidth) {
        h = bandwidth.value();
    }
    else {
        h = interquartile_range(x) * 1.414f;
    }
    if (h <= 0) {
        throw std::invalid_argument("invalid bandwidth");
    }

    bool ok = true; // needed to exit OMP block
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m; i++) {
        ok &= (PyErr_CheckSignals() == 0);
        if (ok) {
            pyi[i] = solve_intercept(px, py, pxi[i], h, n);
        }
    }

    if (!ok) {
        throw py::error_already_set();
    }
    return yi;
}

py::array_t<float> histogram(array_t x, array_t bins, std::optional<float> bandwidth)
{
    verify(x.ndim() == 1 or x.shape(1)==1, "x must be 1-dimensional");

    int n = x.shape(0);
    int m = bins.shape(0);
    py::array_t<float> y = py::array_t<float>(m);

    float       *p_y = y.mutable_data(0);
    const float *p_x = x.data(0);
    const float *p_b = bins.data(0);

    float h;
    if (bandwidth) {
        h = bandwidth.value();
    }
    else {
        float A = interquartile_range(x) / 1.349f;  // Eq (3.3)
        h = 0.9f * A / sqrtf(m);                    // Eq (3.2)
    }
    if (h <= 0) {
        throw std::invalid_argument("invalid bandwidth");
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m; ++i) {
        p_y[i] = histogram_kernel(p_x, p_b[i], h, n);
    }
    return y;
}

PYBIND11_MODULE(lowesslib, m)
{
    m.doc() = "LOWESS: Locally Weighted Scatterplot Smoothing";
    m.def("smooth", &smooth, "Lowess smoothing",
          py::arg("xi"), py::arg("x"), py::arg("y"), py::arg("bandwidth") = py::none());
    m.def("histogram", &histogram,
          "Histogram via kernel density estimation\n\n"
          "See: Chapter 3 of \"Applied Regression Analysis and Generalized Linear Models\"",
          py::arg("x"), py::arg("bins"), py::arg("bandwidth") = py::none());
}
