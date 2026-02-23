#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

/*
 * Force arrays to be contiguous and cast to float. This could create a copy so
 * there might be some room for optimization here later on, perhaps. The reason
 * if because it might actually pay to make the copy so that memory is aligned
 * correctly as we do all these nested loops.
 */
typedef py::array_t<float, py::array::f_style | py::array::forcecast> array_t;
typedef py::array_t<float, py::array::c_style | py::array::forcecast> carray_t;

float solve_intercept(const float *x, const float *y, float x0, float h, int n);
float histogram_kernel(const float *x, float x0, float h, int n);
float interact_kernel(const float *x, const float *y, const float *z, float z0, float h, int n);
void distance(float *u, const float *X, const float *x0, float h, int ldx, int n, int m);

void verify(bool cond, const char *msg)
{
    if (cond == false) {
        throw std::invalid_argument(msg);
    }
}

/*
 * Create a new pybind array by copying the data. If you use the default ctor
 * it will only create a view and can not be returned as output.
 */
array_t new_array_t(int size, const float *data)
{
    array_t out(size);
    std::memcpy(out.mutable_data(), data, size * sizeof(float));
    return out;
}


array_t verify_1d_contiguous(array_t x, const char *label)
{
    char msg[120];
    x = x.squeeze();
    if (x.shape(0) == 0) {
        snprintf(msg, sizeof(msg), "`%s` is empty", label);
        throw std::invalid_argument(msg);
    }
    if (x.ndim() != 1) {
        snprintf(msg, sizeof(msg), "`%s` is not one-dimensional", label);
        throw std::invalid_argument(msg);
    }
    if (x.strides(0) != x.itemsize()) {
        snprintf(msg, sizeof(msg), "`%s` is not contiguous", label);
        throw std::invalid_argument(msg);
    }
    return x;
}

carray_t verify_2d_contiguous(carray_t x, const char *label)
{
    char msg[120];
    if (x.shape(0) == 0) {
        snprintf(msg, sizeof(msg), "`%s` is empty", label);
        throw std::invalid_argument(msg);
    }
    if (x.ndim() < 1 || x.ndim() > 2) {
        snprintf(msg, sizeof(msg), "`%s` is not two-dimensional", label);
        throw std::invalid_argument(msg);
    }

    int n = x.shape(0);
    if (x.ndim() == 1) {
        x = x.reshape({n,1});
    }

    if (x.strides(1) != x.itemsize()) {
        snprintf(msg, sizeof(msg), "`%s` is not C-contiguous", label);
        throw std::invalid_argument(msg);
    }
    return x;
}


/*
 * Sorting can be a major bottleneck for very large arrays. So the simplest
 * approach is to just sub-sample the data and sort the reduced set.
 */
std::vector<float> subsample_sort(const float *x, size_t n)
{
    const size_t MAX_SIZE = 100000;
    const size_t stride   = std::max(1ul, n / MAX_SIZE);

    std::vector<float> y;
    y.reserve((n+stride-1) / stride);

    const float *end = x+n;
    for (const float *p = x; p < end; p += stride) {
        y.push_back(*p);
    }
    std::sort(y.begin(), y.end());
    return y;
}


float interquartile_range(const array_t &x)
{
    std::vector<float> x_sort = subsample_sort(x.data(), x.size());
    const int n = x_sort.size();
    verify(n > 0, "empty vector for range");
    return x_sort[n*3/4] - x_sort[n/4];
}


array_t generate_bins(const float *x, int n, int num_bins)
{
    std::vector<float> sorted_x = subsample_sort(x, n);
    std::vector<float> out(num_bins);
    const float slope = float(sorted_x.size() - 1) / float(num_bins + 1);
    for (int i = 0; i < num_bins; ++i) {
        out[i] = sorted_x[std::round(float(i + 1) * slope)];
    }
    return new_array_t(out.size(), out.data());
}


array_t generate_linear_bins(const float *x, int n, int num_bins)
{
    std::pair minmax = std::minmax_element(x, x+n);
    float y0 = *minmax.first;
    float y1 = *minmax.second;

    std::vector<float> out(num_bins);
    const float slope = (y1 - y0) / float(num_bins - 1);
    for (int i = 0; i < num_bins; ++i) {
        out[i] = y0 + slope * i;
    }
    return new_array_t(out.size(), out.data());
}

/*
 * Take two arrays and drop any rows from both if any have NAN's
 */
std::vector<array_t> drop_any_nans(const std::vector<array_t> &xy)
{
    const int k = xy.size();
    const int n = xy[0].shape(0);
    for (const auto &v: xy) {
        verify(v.shape(0) == n, "input length mismatch");
    }

    std::vector<std::vector<float>> out;
    out.reserve(k);
    for (int j = 0; j < k; ++j) {
        out.emplace_back(n);
    }

    /*
     * TODO - use unchecked views instead of raw pointers
     */
    std::vector<const float*> p;
    for (const auto &v: xy) {
        verify(v.ndim() == 1, "data must be 1-D array");
        verify(v.strides(0) == v.itemsize(), "data not contiguous");
        p.push_back(v.data());
    }

    int row = 0;
    for (int i = 0; i < n; ++i) {
        float sum = 0.0;
        for (int j = 0; j < k; ++j) {
            const float x = p[j][i];
            out[j][row] = x;
            sum += x;
        }
        row += std::isfinite(sum);
    }

    /*
     * TODO - pre-allocate the pybind array and return a slice instead of
     *        creating and copying the data from a temporary vector.
     */
    std::vector<array_t> v_out;
    v_out.reserve(out.size());
    for (const auto &o: out) {
        v_out.push_back(std::move(new_array_t(row, o.data())));
    }
    return v_out;
}


/*
 * If `bins` is a scalar, generate the interpolation points along `x`. Otherwise
 * simply return `bins` after performing some validation.
 *
 * TODO - we're sorting the `x` array twice, one for generating bins and again
 *        for calculating the bandwidth. We should sub-sample by 1/10 for large
 *        arrays and/or look at parallelized versions
 */
array_t process_bins_array(const array_t &x, py::object bins, bool linear=false)
{
    array_t bin_array;
    if (py::isinstance<py::int_>(bins)) {
        const float *p = x.data();
        int n = x.shape(0);
        int m = std::min(bins.cast<int>(), n);
        bin_array = linear? generate_linear_bins(p, n, m) : generate_bins(p, n, m);
    }
    else {
        bin_array = py::array::ensure(bins).squeeze();
        if (!bin_array) {
            throw std::invalid_argument("`bins` cannot be converted to ndarray");
        }
        verify(bin_array.ndim() == 1, "`bins` is not a vector");
    }
    return bin_array;
}


/*
 * Calling python functions from within a thread is a bit of a mess.
 * See: https://github.com/python/cpython/issues/111034
 * See: https://stackoverflow.com/q/78200321
 */
template <typename Func> void parallel_apply(int m, Func && func)
{
    bool ok = true;

    py::gil_scoped_release gil_r;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m; i++) {
        if (i % 32 == 0) {
            py::gil_scoped_acquire gil;
            ok &= (PyErr_CheckSignals() == 0);  // exit loop on CTRL-C
        }

        if (ok) {
            func(i);
        }
    }

    if (!ok) {
        throw py::error_already_set();
    }
}


template <class Bandwidth, class Func>
float unwrap(const Bandwidth& bandwidth, Func&& func)
{
    float h = bandwidth ? *bandwidth : static_cast<float>(func());
    verify(h > 0, "invalid bandwidth");
    return h;
}


//-----------------------------------------------------------------------------


std::tuple<array_t,array_t> smooth(array_t x, array_t y, py::object bins,
                                   std::optional<float> bandwidth,
                                   bool dropna)
{
    x = verify_1d_contiguous(x, "x");
    y = verify_1d_contiguous(y, "y");
    if (dropna) {
        auto out = drop_any_nans({x,y});
        x = std::move(out[0]);
        y = std::move(out[1]);
    }

    array_t x_out = process_bins_array(x, bins);
    const float *pxi = x_out.data(0);
    const float *px  = x.data(0);
    const float *py  = y.data(0);
    const int    m   = x_out.shape(0);
    const int    n   = x.shape(0);

    py::array_t<float> y_out = py::array_t<float>(m);
    float *pyi = y_out.mutable_data(0);

    float h = unwrap(bandwidth, [&]() {
        return interquartile_range(x) * 1.414f;
    });

    parallel_apply(m, [&](int i) {
        pyi[i] = solve_intercept(px, py, pxi[i], h, n);
    });

    return std::make_tuple(x_out, y_out);
}


array_t forecast(carray_t x, array_t y, carray_t xi, float bandwidth)
{
    y  = verify_1d_contiguous(y,  "y");
    x  = verify_2d_contiguous(x,  "x");
    xi = verify_2d_contiguous(xi, "xi");
    verify(x.shape(0) == y.shape(0),  "inconsistent dimensions: x and y");
    verify(x.shape(1) == xi.shape(1), "inconsistent dimensions: x and xi");
    verify(bandwidth > 0, "invalid bandwidth");

    const int n = x.shape(0);
    const int m = x.shape(1);
    const int ldx  = x.strides(0)  / x.itemsize();
    const int ldxi = xi.strides(0) / xi.itemsize();
    py::array_t<float> yi(n);
    std::vector<float> u(n);

    const float *py = y.data(0);
    float *pyi = yi.mutable_data(0);

    for (int j = 0; j < xi.shape(0); ++j) {
        distance(u.data(), x.data(0), xi.data(ldxi*j), bandwidth, ldx, n, m);

        float x00 = 0;
        float x01 = 0;
        float x11 = 0;
        float xy0 = 0;
        float xy1 = 0;
        for (int i = 0; i < n; ++i) {
            float w = expf(-0.5 * u[i] * u[i]);
            float ww = w * w;
            x00 += ww;
            x01 += ww * u[i];
            x11 += ww * u[i] * u[i];
            xy0 += ww * py[i];
            xy1 += ww * u[i] * py[i];
        }

        float numer = x11 * xy0 - x01 * xy1;
        float denom = x00 * x11 - x01 * x01;
        pyi[j] = denom > 0? numer / denom : 0.0f;
    }

    return yi;
}


std::tuple<array_t,array_t> histogram(array_t x, py::object bins,
                                      std::optional<float> bandwidth,
                                      bool dropna)
{
    x = verify_1d_contiguous(x, "x");
    if (dropna) {
        x = std::move(drop_any_nans({x})[0]);
    }

    array_t bin_array = process_bins_array(x, bins, true);
    int n = x.shape(0);
    int m = bin_array.shape(0);
    py::array_t<float> y = py::array_t<float>(m);

    float       *p_y = y.mutable_data(0);
    const float *p_x = x.data(0);
    const float *p_b = bin_array.data(0);

    float h = unwrap(bandwidth, [&]() {
        float A = interquartile_range(x) / 1.349f;  // Eq (3.3)
        float h = 0.9f * A / sqrtf(m);              // Eq (3.2)
        if (h == 0) {
            auto [min, max] = std::minmax_element(x.data(), x.data()+x.size());
            h = (*max - *min) * 0.1; // desperate guess
        }
        return h;
    });

    parallel_apply(m, [&](int i) {
        p_y[i] = histogram_kernel(p_x, p_b[i], h, n);
    });

    return std::make_tuple(bin_array, y);
}


std::tuple<array_t,array_t> interact(array_t x, array_t y, array_t z, py::object bins,
                                     std::optional<float> bandwidth, bool dropna)
{
    x = verify_1d_contiguous(x, "x");
    y = verify_1d_contiguous(y, "y");
    z = verify_1d_contiguous(z, "z");
    if (dropna) {
        auto out = drop_any_nans({x,y,z});
        x = std::move(out[0]);
        y = std::move(out[1]);
        z = std::move(out[2]);
    }

    array_t zi = process_bins_array(z, bins);
    const int n = x.shape(0);
    const int m = zi.shape(0);

    float h = unwrap(bandwidth, [&]() {
        return interquartile_range(z) * 0.2f; // not sure what value to use here
    });

    const float *p_x  = x.data(0);
    const float *p_y  = y.data(0);
    const float *p_z  = z.data(0);
    const float *p_zi = zi.data(0);

    py::array_t<float> bi = py::array_t<float>(m);
    float *p_bi = bi.mutable_data(0);

    parallel_apply(m, [&](int i) {
        p_bi[i] = interact_kernel(p_x, p_y, p_z, p_zi[i], h, n);
    });

    return std::make_tuple(zi, bi);
}


PYBIND11_MODULE(lowesslib, m)
{
    py::options options;
    options.disable_function_signatures(); // Disable auto-generated

    m.doc() = "LOWESS: Locally Weighted Scatter plot Smoothing";

    m.def("smooth", &smooth,
          R"pbdoc(smooth(x, y, bins=100, bandwidth=None, dropna=True)

    Perform LOWESS (Locally Weighted Scatterplot Smoothing) on one-dimensional data.

    Parameters
    ----------
    x : array_like, shape (N,)
        Independent variable.

    y : array_like, shape (N,)
        Dependent (response) variable.

    bins : int or sequence of scalars, optional
        If an integer, specifies the number of quantiles in `x` to use as
        interpolation points. If a sequence, it is treated as the exact
        interpolation point locations.

    bandwidth : float, optional
        Kernel bandwidth for smoothing, in the same units as `x`.
        Defaults to 1.41 times the interquartile range (IQR) of `x`.

    dropna: bool, optional
        Remove all NAN's and INF's from the data. This will make a copy.

    Returns
    -------
    xi : ndarray
        Interpolation point locations in `x`.

    yi : ndarray
        Smoothed, interpolated values of `y` at `xi`.)pbdoc",

          py::arg("x"), py::arg("y"), py::arg("bins") = 100,
          py::arg("bandwidth") = py::none(),
          py::arg("dropna") = true);

    //-------------------------------------------------------------------------

    m.def("forecast", &forecast,
          R"pbdoc(forecast(x, y, xo, bandwidth))pbdoc",
          py::arg("x"), py::arg("y"), py::arg("xo"), py::arg("bandwidth"));

    //-------------------------------------------------------------------------

    m.def("histogram", &histogram,
          R"pbdoc(histogram(x, bins=100, bandwidth=None, dropna=True)

    Estimate the histogram of a dataset using kernel density estimation.

    Parameters
    ----------
    x : array_like, shape (N,)
        Input data. The histogram is computed over the entire array.

    bins : int or sequence of scalars
        If an integer, specifies the number of bins to use. If a sequence, it
        should be a monotonically increasing array of bin center locations.

    bandwidth : float, optional
        Kernel bandwidth for smoothing, in the same units as `x`.

    dropna: bool, optional
        Remove all NAN's and INF's from the data. This will make a copy.

    Returns
    -------
    bins : ndarray
        Bin center locations.

    density : ndarray
        Estimated probability *density* function values at the bin locations.

    See Also
    --------
    Chapter 3 of "Applied Regression Analysis and Generalized Linear Models")pbdoc",

          py::arg("x"), py::arg("bins") = 100, py::arg("bandwidth") = py::none(),
          py::arg("dropna") = true);

    //-------------------------------------------------------------------------

    m.def("interact", &interact,
          R"pbdoc(interact(x, y, z, bins=100, bandwidth=None, dropna=True)

    Identify the functional form of `z` in the interaction model: `y = x * f(z)`

    Parameters
    ----------
    x : array_like, shape (N,)
        Independent variable.

    y : array_like, shape (N,)
        Dependent variable, assumed to be a noisy function of the interaction
        between `x` and `z`.

    z : array_like, shape (N,)
        Nonlinear interaction term.

    bins : int or sequence of scalars, optional
        If an integer, specifies the number of quantiles in `z` to use as
        interpolation points. If a sequence, it is treated as the exact
        interpolation point locations.

    bandwidth : float, optional
        Kernel bandwidth for smoothing, in the same units as `z`.

    dropna: bool, optional
        Remove all NAN's and INF's from the data. This will make a copy.

    Returns
    -------
    zi : ndarray
        Interpolation point locations in `z`.

    f(zi) : ndarray
        Estimated functional form of `f(z)` in the model `y = x * f(z)`)pbdoc",
          py::arg("x"), py::arg("y"), py::arg("z"), py::arg("bins") = 100,
          py::arg("bandwidth") = py::none(),
          py::arg("dropna") = true);

}
