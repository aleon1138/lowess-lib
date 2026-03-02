import unittest
import timeit
import numpy as np
import sys

sys.path.insert(0, "ext")
import lowesslib_numba as low1
import lowesslib as low2

np.random.seed(42)


def generate_data(n):
    x = np.random.randn(n).astype("f")
    z = np.abs(np.random.randn(n)).astype("f")
    y = x * z + np.random.randn(n).astype("f")
    return x, y, z


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
