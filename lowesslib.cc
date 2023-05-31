#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

extern "C" {
    float solve_intercept(const float *x, const float *y, float x0, float h, int n);
}

py::array_t<float> smooth(
    py::array_t<float, py::array::f_style | py::array::forcecast> xi,
    py::array_t<float, py::array::f_style | py::array::forcecast> x,
    py::array_t<float, py::array::f_style | py::array::forcecast> y,
    float h
)
{
    if (x.shape(0) != y.shape(0)) {
        throw std::invalid_argument("x and y are not of the same length");
    }
    if (x.shape(0) == 0) {
        throw std::invalid_argument("array of sample points is empty");
    }
    if (xi.ndim() > 1) {
        throw std::invalid_argument("xi must be 1-dimensional");
    }
    if (x.ndim() > 1) {
        throw std::invalid_argument("x must be 1-dimensional");
    }
    if (y.ndim() > 1) {
        throw std::invalid_argument("y must be 1-dimensional");
    }

    auto yi = py::array_t<float>(xi.shape(0));
    auto x_buffer  = x.request();
    auto y_buffer  = y.request();
    auto xi_buffer = xi.request();
    auto yi_buffer = yi.request(true);

    float *pxi = static_cast<float*>(xi_buffer.ptr);
    float *pyi = static_cast<float*>(yi_buffer.ptr);
    float *px  = static_cast<float*>(x_buffer.ptr);
    float *py  = static_cast<float*>(y_buffer.ptr);
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
    py::options options;
    options.disable_function_signatures();

    m.doc() = "LOWESS: Locally Weighted Scatterplot Smoothing";
    m.def("smooth", &smooth, "Lowess smoothing",
          py::arg("xi"), py::arg("x"), py::arg("y"), py::arg("h"));
}
