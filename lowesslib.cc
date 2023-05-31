#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

extern "C" {
    float solve_intercept(const float *x, const float *y, float x0, float h, int n);
}

void verify(bool cond, const char*msg)
{
    if (cond == false) {
        throw std::invalid_argument(msg);
    }
}

py::array_t<float> smooth(
    py::array_t<float, py::array::f_style | py::array::forcecast> xi,
    py::array_t<float, py::array::f_style | py::array::forcecast> x,
    py::array_t<float, py::array::f_style | py::array::forcecast> y,
    float h
)
{
    verify(x.shape(0) == y.shape(0), "x and y are not of the same length");
    verify(x.shape(0) > 0, "x is empty");
    verify(xi.ndim() == 1, "xi must be 1-dimensional");
    verify(x.ndim()  == 1, "x must be 1-dimensional");
    verify(y.ndim()  == 1, "y must be 1-dimensional");

    auto yi = py::array_t<float>(xi.shape(0));
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

PYBIND11_MODULE(lowesslib, m)
{
    m.doc() = "LOWESS: Locally Weighted Scatterplot Smoothing";
    m.def("smooth", &smooth, "Lowess smoothing",
          py::arg("xi"), py::arg("x"), py::arg("y"), py::arg("h"));
}
