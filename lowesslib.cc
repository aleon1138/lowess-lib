#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
typedef py::array_t<float, py::array::f_style | py::array::forcecast> array_t;

extern "C" {
    float solve_intercept(const float *x, const float *y, float x0, float h, int n);
    float histogram_kernel(const float *x, float x0, float h, int n);
}

/*
 *  Use interquartile range to get a default kernel bandwidth. We use a rough
 *  approximation via gradient descent.
 */
float default_bandwidth(const float *x, int n, int nbins)
{
    float q25(0), q75(0);
    float w = 1.0f/float(n);
    #pragma omp parallel for reduction(+:q25,q75) schedule(static)
    for (int i = 0; i < n; ++i) {
        q25 += w * ((q25 > x[i])? -0.75f : 0.25f);
        q75 += w * ((q75 > x[i])? -0.25f : 0.75f);
    }
    float A = (q75 - q25) / 1.349f;  // Eq (3.3)
    return 0.9f * A / sqrtf(nbins);  // Eq (3.2)
}

void verify(bool cond, const char*msg)
{
    if (cond == false) {
        throw std::invalid_argument(msg);
    }
}

py::array_t<float> smooth(array_t xi, array_t x, array_t y, float h)
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
    const float *p_x = x.mutable_data(0);
    const float *p_b = bins.mutable_data(0);
    float h = bandwidth? bandwidth.value() : default_bandwidth(p_x, n, m);

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
          py::arg("xi"), py::arg("x"), py::arg("y"), py::arg("h"));
    m.def("histogram", &histogram,
          "Histogram via kernel density estimation\n\n"
          "See: Chapter 3 of \"Applied Regression Analysis and Generalized Linear Models\"",
          py::arg("x"), py::arg("bins"), py::arg("bandwidth") = py::none());
}
