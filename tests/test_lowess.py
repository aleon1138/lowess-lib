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
            bi_ref[i] = xy / xx if xx > 0 else 0

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
