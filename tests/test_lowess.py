import os
import unittest
import timeit
import scipy.optimize
import numpy as np
import sys

_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(_root, "ext"))
import lowesslib_numba as low1
import lowesslib as low2

np.random.seed(42)


def generate_data(n):
    x = np.random.randn(n).astype("f")
    z = np.abs(np.random.randn(n)).astype("f")
    y = x * z + np.random.randn(n).astype("f")
    return x, y, z


def lowess_quantile_1(x, y, bins, tau, h):

    def _loss(theta, x, y, tau):
        w = np.exp(-0.5 * np.square(x))
        e = theta[0] + theta[1] * x - y
        tau = np.where(e >= 0.0, 1.0 - tau, tau)
        return (tau * np.square(e * w)).mean()

    yi = np.zeros(len(bins))
    for i in range(len(bins)):
        res = scipy.optimize.minimize(
            _loss,
            x0=np.zeros(2),
            args=((x - bins[i]) / h, y, tau),
            method="Nelder-Mead",
        )
        yi[i] = res.x[0]
    return bins, yi


def lowess_quantile_2(x, y, bins, tau, h):
    return low2.expectile(x, y, tau, bins, h)


class TestLowess(unittest.TestCase):

    def test_smooth_avx_tail(self):
        x, y, z = generate_data(7)

        a = low1.smooth(x, y, bins=x, bandwidth=2.0)
        b = low2.smooth(x, y, bins=x, bandwidth=2.0)

        self.assertTrue((a[1] - b[1]).std() < 1e-6)

    def test_smooth_avx(self):
        x, y, z = generate_data(8 * 1252)

        a = low1.smooth(x, y)
        b = low2.smooth(x, y)

        # AVX implementation for `exp(x)` is only an approximation
        self.assertTrue((a[0] - b[0]).std() < 1e-9)
        self.assertTrue((a[1] - b[1]).std() < 1e-2)

    def test_smooth_sparse_tail(self):
        """
        Bins extending past the data force the local-linear system towards
        degeneracy: `x01*x01` approaches `x00*x11` and the denominator is left
        with only a few significant digits. Accumulating the normal equations
        in float used to lose those digits, and the tail of the fit collapsed
        into noise (values several times larger than max(y), or a hard 0).
        """
        n = 200_000
        x = np.clip(np.random.beta(8, 2, n) * 0.15 + 0.85, 0, 0.962).astype("f")
        y = np.random.uniform(0, 0.989, n).astype("f")
        y[np.random.rand(n) < 0.3] = np.nan
        bins = np.linspace(0.9, 1.0, 100)
        h = 0.01

        # Same estimator, accumulated in double, with the same guards
        m = np.isfinite(x) & np.isfinite(y)
        xd, yd = x[m].astype("f8"), y[m].astype("f8")
        ref = np.full(len(bins), np.nan)
        for i, x0 in enumerate(bins):
            u = (x0 - xd) / h
            w2 = np.where(u * u < 2 * low1.GAUSS_CUTOFF, np.exp(-u * u), 0.0)
            x00, x01, x11 = w2.sum(), (w2 * u).sum(), (w2 * u * u).sum()
            xy0, xy1 = (w2 * yd).sum(), (w2 * u * yd).sum()
            if x00 <= 0 or abs(x01 / x00) > low1.MAX_EXTRAPOLATION:
                continue
            denom = x00 * x11 - x01 * x01
            if denom > low1.COND_TOL * x00 * x11:
                ref[i] = (x11 * xy0 - x01 * xy1) / denom

        _, yi = low2.smooth(x, y, bins, bandwidth=h)
        self.assertTrue(np.array_equal(np.isnan(yi), np.isnan(ref)))
        self.assertTrue(np.nanmax(np.abs(yi - ref)) < 1e-2)

        # A local linear fit of data bounded by [0, 0.989] has no business
        # returning 3.4 anywhere on this grid.
        yi = yi[np.isfinite(yi)]
        self.assertTrue(yi.min() > -0.1 and yi.max() < 1.1)

    def test_degenerate_fit_is_nan(self):
        """
        A local window with no spread in `x` cannot identify a slope. That has
        to come back as NaN, not 0 — 0 is a plausible estimate and would blend
        into the curve unnoticed.
        """
        y = np.random.rand(1000).astype("f")

        # Every point at the same location, so the weighted variance of u is 0
        x = np.full(1000, 0.5, dtype="f")
        _, yi = low2.smooth(x, y, np.linspace(0.4, 0.6, 5), bandwidth=0.01)
        self.assertTrue(np.isnan(yi).all())

        # `interact` uses the same sentinel: x is identically zero, so there is
        # no scale to estimate
        z = np.random.rand(1000).astype("f")
        _, bi = low2.interact(
            np.zeros(1000, "f"), y, z, np.linspace(0.2, 0.8, 4), bandwidth=0.1
        )
        self.assertTrue(np.isnan(bi).all())

        # A well-posed fit must stay finite
        x = np.random.randn(1000).astype("f")
        _, yi = low2.smooth(x, x * 2 + y, np.linspace(-1, 1, 20), bandwidth=0.5)
        self.assertTrue(np.isfinite(yi).all())

    def test_expectile_is_scale_equivariant(self):
        """
        An expectile satisfies `expectile(c*y) == c*expectile(y)`, so rescaling
        `y` must be a no-op. The previous Nelder-Mead solver stopped on the
        absolute spread of the loss, and since the loss was divided by the full
        point count rather than the weight the window carried, its magnitude
        collapsed on sparse windows and the solver quit early. Rescaling `y`
        then changed the answer, which is how that bug shows itself.
        """
        n = 100_000
        x = np.random.rand(n).astype("f")
        y = np.clip(0.3 + 0.5 * x + 0.05 * np.random.randn(n), 0, 1).astype("f")
        y[x > 0.85] = np.nan
        bins = np.linspace(0.70, 0.88, 25)

        _, base = low2.expectile(x, y, 0.1, bins, bandwidth=0.01)
        for c in (1e-3, 1e3):
            _, s = low2.expectile(x, (y * c).astype("f"), 0.1, bins, bandwidth=0.01)
            # equal_nan also asserts that which bins are refused does not
            # depend on the scale of `y`
            self.assertTrue(
                np.allclose(s / c, base, rtol=1e-4, atol=1e-6, equal_nan=True)
            )

    def test_expectile_monotone_in_tau(self):
        """
        Expectiles are non-decreasing in `tau`, by definition. Checked over
        windows that run from the interior out to the edge of the data, since
        that is where a solver that quits early goes wrong: each `tau` bails
        out at a different point and the answers come back out of order.
        """
        n = 100_000
        x = np.random.rand(n).astype("f")
        y = np.clip(0.3 + 0.5 * x + 0.05 * np.random.randn(n), 0, 1).astype("f")
        y[x > 0.85] = np.nan
        bins = np.linspace(0.70, 0.87, 25)

        taus = np.arange(0.1, 1.0, 0.1)
        e = np.array(
            [low2.expectile(x, y, t, bins, bandwidth=0.01)[1] for t in taus]
        )
        # which bins are refused must not depend on tau
        self.assertTrue((np.isnan(e) == np.isnan(e[0])).all())
        e = e[:, np.isfinite(e[0])]
        self.assertTrue(e.size > 0)
        self.assertTrue((np.diff(e, axis=0) >= -1e-5).all())

    def test_expectile_half_matches_smooth(self):
        """
        At `tau = 0.5` the asymmetric weight is constant, so the expectile is
        the ordinary local linear fit. The two differ only in that `smooth()`
        weights with the fast SIMD exp approximation.
        """
        n = 50_000
        x = np.random.randn(n).astype("f")
        y = (x + np.random.randn(n)).astype("f")
        bins = np.linspace(-1.5, 1.5, 20)

        _, a = low2.smooth(x, y, bins, bandwidth=0.3)
        _, b = low2.expectile(x, y, 0.5, bins, bandwidth=0.3)
        self.assertTrue(np.abs(a - b).max() < 1e-2)

    def test_extrapolation_is_nan(self):
        """
        A local linear fit reports where the fitted line crosses the evaluation
        point, i.e. `ybar - b*ubar`. Once the supporting data has drifted a few
        bandwidths away, `b` is fitting noise and `ubar` amplifies it, so the
        result can be many times larger than any real `y`. Those bins must come
        back as NaN rather than as a confident-looking number.
        """
        n = 200_000
        x = np.random.rand(n).astype("f")
        y = np.clip(0.3 + 0.5 * x + 0.05 * np.random.randn(n), 0, 1).astype("f")
        y[x > 0.85] = np.nan  # no data survives above 0.85
        h = 0.01

        # Entirely past the data: nothing here is estimable
        _, yi = low2.smooth(x, y, np.linspace(0.9, 1.0, 50), bandwidth=h)
        self.assertTrue(np.isnan(yi).all())

        # Interior bins, and the one-sided window right at the data edge, must
        # still be fit normally and accurately
        bins = np.linspace(0.0, 0.85, 20)
        _, yi = low2.smooth(x, y, bins, bandwidth=h)
        self.assertTrue(np.isfinite(yi).all())
        self.assertTrue(np.abs(yi - (0.3 + 0.5 * bins)).max() < 1e-2)

    def test_expectile_extrapolation_is_nan(self):
        """
        `solve_expectile` fits `theta[0] + theta[1]*u` and reports `theta[0]`,
        the value at the evaluation point, so it has the same lever arm as
        `smooth()` and must refuse the same windows. The support test is taken
        under the symmetric kernel weights, so the decision does not depend on
        `tau` — check that it holds at strongly asymmetric levels too.
        """
        n = 200_000
        x = np.random.rand(n).astype("f")
        y = np.clip(0.3 + 0.5 * x + 0.05 * np.random.randn(n), 0, 1).astype("f")
        y[x > 0.85] = np.nan  # no data survives above 0.85
        h = 0.01

        far = np.linspace(0.9, 1.0, 20)
        near = np.linspace(0.0, 0.85, 20)
        for tau in (0.1, 0.5, 0.9):
            # Entirely past the data: nothing here is estimable
            _, yi = low2.expectile(x, y, tau, far, bandwidth=h)
            self.assertTrue(np.isnan(yi).all(), f"tau={tau}")

            # Interior bins, and the one-sided window at the data edge, must
            # still be fit -- the guard must not refuse them at any tau
            _, yi = low2.expectile(x, y, tau, near, bandwidth=h)
            self.assertTrue(np.isfinite(yi).all(), f"tau={tau}")

            # and they must stay ordered in tau, and near the conditional mean
            self.assertTrue(np.abs(yi - (0.3 + 0.5 * near)).max() < 0.1, f"tau={tau}")

    def test_interact(self):
        x, y, z = generate_data(8 * 1252 + 3)  # not a multiple of 8, exercises scalar tail
        bins = np.linspace(0.1, 2.0, 20).astype("f")
        h = 0.3

        # Python reference implementation
        bi_ref = np.zeros(len(bins), dtype="f")
        for i, z0 in enumerate(bins):
            w = np.exp(-0.5 * ((z0 - z) / h) ** 2).astype("f")
            w2 = w * w
            xx = np.sum(x * x * w2)
            xy = np.sum(x * y * w2)
            bi_ref[i] = xy / xx if xx > 0 else np.nan

        zi, bi = low2.interact(x, y, z, bins=bins, bandwidth=h)

        self.assertTrue(np.allclose(zi, bins))
        self.assertTrue((bi - bi_ref).std() < 1e-2)

    def test_expectile(self):
        x = np.random.randn(8 * 1234 + 7)
        y = np.maximum(-x, 0) + np.random.randn(len(x)) * np.maximum(x / 2, 0.2)
        x = x.astype("f")
        y = y.astype("f")

        bins = np.linspace(-2, 2, 20)
        a = lowess_quantile_1(x, y, bins, 0.75, 0.1)
        b = lowess_quantile_2(x, y, bins, 0.75, 0.1)
        self.assertTrue((a[1] - b[1]).std() < 1e-3)


def benchmark():
    ns = [
        1_000,
        2_000,
        5_000,
        10_000,
        20_000,
        50_000,
        100_000,
        200_000,
        500_000,
        1_000_000,
        2_000_000,
        5_000_000,
        10_000_000,
    ]
    x, y, z = generate_data(10_000_000)
    for n in ns:
        x_, y_ = x[:n], y[:n]
        dt1 = timeit.repeat(lambda: low1.smooth(x_, y_), number=1, repeat=10)
        dt2 = timeit.repeat(
            lambda: low2.smooth(x_, y_, dropna=False), number=1, repeat=10
        )
        print(f"{n},{min(dt1)*1e3:.2f},{min(dt2)*1e3:.2f}")


if __name__ == "__main__":
    unittest.main()
