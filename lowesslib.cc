#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <omp.h>

namespace py = pybind11;

// Force arrays to be column-contiguous and cast to float
typedef py::array_t<float, py::array::f_style | py::array::forcecast> array_t;

float solve_intercept(const float *x, const float *y, float x0, float h, int n);
float histogram_kernel(const float *x, float x0, float h, int n);
float interact_kernel(const float *x, const float *y, const float *z, float z0, float h, int n);

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


void verify(bool cond, const char *msg)
{
    if (cond == false) {
        throw std::invalid_argument(msg);
    }
}


array_t verify_1d(array_t x, const char *label)
{
    char msg[80];
    x = x.squeeze();
    if (x.shape(0) == 0) {
        sprintf(msg, "`%s` is empty", label);
        throw std::invalid_argument(msg);
    }
    if (x.ndim() != 1) {
        sprintf(msg, "`%s` is not one-dimensional", label);
        throw std::invalid_argument(msg);
    }
    return x;
}


array_t generate_bins(const array_t x, int bins)
{
    py::buffer_info info = x.request();
    int n = info.size;
    bins  = std::min(bins, n);

    float* ptr = static_cast<float*>(info.ptr);
    std::vector<float> sorted_x(ptr, ptr + n);
    std::sort(sorted_x.begin(), sorted_x.end());

    std::vector<float> out(bins);
    const float slope = float(n - 1) / float(bins + 1);
    for (int i = 0; i < bins; ++i) {
        out[i] = sorted_x[std::round(float(i + 1) * slope)];
    }
    return array_t(out.size(), out.data());
}


/*
 * Handle creation or conversion of `bin_array`.
 *
 * TODO - we're sorting the `x` array twice, one for generating bins and again
 *        for calculating the bandwidth. We should sub-sample by 1/10 for large
 *        arrays and/or look at parallelized versions
 */
array_t process_bins_array(const array_t x, py::object bins)
{
    array_t bin_array;
    if (py::isinstance<py::int_>(bins)) {
        bin_array = generate_bins(x, bins.cast<int>());
    }
    else {
        bin_array = py::array::ensure(bins);
        if (!bin_array) {
            throw std::invalid_argument("`bins` cannot be converted to ndarray");
        }
        verify(bin_array.ndim() == 1, "`bins` is not a vector");
    }
    return bin_array;
}


std::tuple<array_t,array_t> smooth(array_t x, array_t y, py::object bins, std::optional<float> bandwidth)
{
    x = verify_1d(x, "x");
    y = verify_1d(y, "y");
    verify(x.shape(0) == y.shape(0), "input length mismatch");

    array_t bin_array = process_bins_array(x, bins);
    const float *pxi = bin_array.data(0);
    const float *px  = x.data(0);
    const float *py  = y.data(0);
    const int    m   = bin_array.shape(0);
    const int    n   = x.shape(0);

    py::array_t<float> yi = py::array_t<float>(m);
    float *pyi = yi.mutable_data(0);

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

    /*
     * Calling python functions from within a thread is a bit of a mess.
     * See: https://github.com/python/cpython/issues/111034
     * See: https://stackoverflow.com/q/78200321
     */
    bool ok = true;
    Py_BEGIN_ALLOW_THREADS  // Releases GIL
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m; i++) {
        if (i % 32 == 0) { // unintelligent optimization
            py::gil_scoped_acquire gil;
            ok &= (PyErr_CheckSignals() == 0);  // exit loop on CTRL-C
        }

        if (ok) {
            pyi[i] = solve_intercept(px, py, pxi[i], h, n);
        }
    }
    Py_END_ALLOW_THREADS  // Re-acquires GIL

    if (!ok) {
        throw py::error_already_set();
    }
    return std::make_tuple(bin_array, yi);
}


std::tuple<array_t,array_t> histogram(array_t x, py::object bins, std::optional<float> bandwidth)
{
    x = verify_1d(x, "x");

    // TODO - create equally-space bins for histogram
    array_t bin_array = process_bins_array(x, bins);
    int n = x.shape(0);
    int m = bin_array.shape(0);
    py::array_t<float> y = py::array_t<float>(m);

    float       *p_y = y.mutable_data(0);
    const float *p_x = x.data(0);
    const float *p_b = bin_array.data(0);

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

    Py_BEGIN_ALLOW_THREADS  // Releases GIL
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m; ++i) {
        p_y[i] = histogram_kernel(p_x, p_b[i], h, n);
    }
    Py_END_ALLOW_THREADS  // Re-acquires GIL

    return std::make_tuple(bin_array, y);
}


std::tuple<array_t,array_t> interact(array_t x, array_t y, array_t z, py::object bins, std::optional<float> bandwidth)
{
    x = verify_1d(x, "x");
    y = verify_1d(y, "y");
    z = verify_1d(z, "z");
    verify(x.shape(0) == y.shape(0), "input length mismatch");
    verify(x.shape(0) == z.shape(0), "input length mismatch");

    array_t zi = process_bins_array(z, bins);
    py::array_t<float> bi = py::array_t<float>(zi.shape(0));

    float h;
    if (bandwidth) {
        h = bandwidth.value();
    }
    else {
        h = interquartile_range(z) * 0.2f; // not sure what value to use here
    }
    if (h <= 0) {
        throw std::invalid_argument("invalid bandwidth");
    }

    int n = x.shape(0);
    int m = zi.shape(0);
    const float *p_x  = x.data(0);
    const float *p_y  = y.data(0);
    const float *p_z  = z.data(0);
    const float *p_zi = zi.data(0);
    float *p_bi = bi.mutable_data(0);

    Py_BEGIN_ALLOW_THREADS  // Releases GIL
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m; ++i) {
        p_bi[i] = interact_kernel(p_x, p_y, p_z, p_zi[i], h, n);
    }
    Py_END_ALLOW_THREADS  // Re-acquires GIL

    return std::make_tuple(zi, bi);
}


PYBIND11_MODULE(lowesslib, m)
{
    py::options options;
    options.disable_function_signatures(); // Disable auto-generated

    m.doc() = "LOWESS: Locally Weighted Scatter plot Smoothing";

    m.def("smooth", &smooth,
          R"pbdoc(smooth(x, y, bins=100, bandwidth=None)

    Perform LOWESS smoothing on one-dimensional data.

    Parameters
    ----------
    x : array_like, shape (N,)
        Independent variable.
    y : array_like, shape (N,)
        Dependent (response) variable.
    bins : int or sequence of scalars, optional
        If an integer, `bins` specifies the number of quantiles in `x` to use
        as interpolation points. If a sequence, it is treated as the exact
        locations of the interpolation points.
    bandwidth : float, optional
        Kernel bandwidth for smoothing, in the same units as `x`.
        The default is 1.41 times the interquartile range of `x`.

    Returns
    -------
    xi : ndarray
        Interpolation point locations.
    yi : ndarray
        Smoothed interpolated values of `y` at `xi`.)pbdoc",

          py::arg("x"), py::arg("y"), py::arg("bins") = 100, py::arg("bandwidth") = py::none());

    //-------------------------------------------------------------------------

    m.def("histogram", &histogram,
          R"pbdoc(histogram(x, bins, bandwidth=None)

    Compute the histogram of a dataset via kernel density estimation.

    Parameters
    ----------
    x : array_like, shape (N,)
        Input data. The histogram is computed over the entire array.
    bins : int or sequence of scalars, optional
        If an integer, `bins` specifies the number of bins to use. If a sequence,
        a monotonically increasing array of bin centre locations.
    bandwidth : float, optional
        Kernel bandwidth for smoothing, in the same units as `x`.

    Returns
    -------
    bins : ndarray
        The bins centre locations.
    density : ndarray
        The probability *density* function at the bin location.

    See Also
    --------
    Chapter 3 of "Applied Regression Analysis and Generalized Linear Models")pbdoc",

          py::arg("x"), py::arg("bins") = 100, py::arg("bandwidth") = py::none());

    //-------------------------------------------------------------------------

    m.def("interact", &interact,
          R"pbdoc(interact(x, y, z, bins=100, bandwidth=None))pbdoc",
          py::arg("x"), py::arg("y"), py::arg("z"), py::arg("bins") = 100, py::arg("bandwidth") = py::none());
}
