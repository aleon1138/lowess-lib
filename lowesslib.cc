#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

extern "C" {
    void lowess_smooth(float *y_out, const float *xi, int m,
                       const float *x, const float *y, int n,
                       float h);
}

void ensure_1d_contiguous(py::array_t<float> &array)
{
    if (array.ndim() > 1) {
        throw std::runtime_error("input must be 1-dimensional");
    }
    if (array.strides(0) != array.itemsize()) {
        throw std::runtime_error("input is not contiguous");
    }
}

py::array_t<float> smooth(
    py::array_t<float> xi,
    py::array_t<float> x,
    py::array_t<float> y,
    float h
)
{
    if (x.shape(0) != y.shape(0)) {
        throw std::runtime_error("x and y are not of the same length");
    }
    if (x.shape(0) == 0) {
        throw std::runtime_error("array of sample points is empty");
    }
    ensure_1d_contiguous(x);
    ensure_1d_contiguous(y);
    ensure_1d_contiguous(xi);

    auto x_buffer  = x.request();
    auto y_buffer  = y.request();
    auto xi_buffer = xi.request();

    auto yi = py::array_t<float>(xi.shape(0));
    auto yi_buffer = yi.request(true);

    lowess_smooth(static_cast<float*>(yi_buffer.ptr),
                  static_cast<float*>(xi_buffer.ptr),
                  xi.shape(0),
                  static_cast<float*>(x_buffer.ptr),
                  static_cast<float*>(y_buffer.ptr),
                  x.shape(0), h);
    return yi;
}

PYBIND11_MODULE(lowesslib, m)
{
    py::options options;
    options.disable_function_signatures();

    m.doc() = "LOWESS: Locally Weighted Scatterplot Smoothing";
    m.def("smooth", &smooth, "Lowess smoothing");
}
