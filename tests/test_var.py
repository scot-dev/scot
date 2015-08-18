# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013-2015 SCoT Development Team

import unittest

import numpy as np

from scot.varbase import VARBase as VAR
from scot.utils import acm

epsilon = 1e-10


class TestVAR(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def generate_data(self, cc=((1, 0), (0, 1))):
        var = VAR(2)
        var.coef = np.array([[0.2, 0.1, 0.4, -0.1], [0.3, -0.2, 0.1, 0]])
        l = (1000, 100)
        x = var.simulate(l, lambda: np.random.randn(2).dot(cc))
        self.assertEqual(x.shape, (l[1], 2, l[0]))
        return x, var

    def test_simulate(self):
        noisefunc = lambda: [1, 1]   # use deterministic function instead of noise
        num_samples = 100

        b = np.array([[0.2, 0.1, 0.4, -0.1], [0.3, -0.2, 0.1, 0]])

        var = VAR(2)
        var.coef = b

        np.random.seed(42)
        x = var.simulate(num_samples, noisefunc)
        self.assertEqual(x.shape, (1, b.shape[0], num_samples))

        # make sure we got expected values within reasonable accuracy
        for n in range(10, num_samples):
            self.assertTrue(np.all(
                np.abs(x[0, :, n] - 1
                       - np.dot(x[0, :, n - 1], b[:, 0::2].T)
                       - np.dot(x[0, :, n - 2], b[:, 1::2].T)) < 1e-10))

    def test_predict(self):
        np.random.seed(777)
        x, var = self.generate_data()
        z = var.predict(x)

        self.assertTrue(np.abs(np.var(x[:, :, 100:] - z[:, :, 100:]) - 1) < 0.005)

    def test_yulewalker(self):
        np.random.seed(7353)
        x, var0 = self.generate_data([[1, 2], [3, 4]])

        acms = [acm(x, l) for l in range(var0.p+1)]

        var = VAR(var0.p)
        var.from_yw(acms)

        # that limit is rather generous, but we don't want tests to fail due to random variation
        self.assertTrue(np.all(np.abs(var0.coef - var.coef) < 0.02))
        self.assertTrue(np.all(np.abs(var0.rescov - var.rescov) < 0.02))

    def test_whiteness(self):
        np.random.seed(91)
        r = np.random.randn(10, 5, 100)     # gaussian white noise
        r0 = r.copy()

        var = VAR(0)
        var.residuals = r

        p = var.test_whiteness(20)

        self.assertTrue(np.all(r == r0))    # make sure we don't modify the input
        self.assertGreater(p, 0.01)         # test should be non-significant for white noise

        r[:,1,3:] = r[:,0,:-3]              # create cross-correlation at lag 3
        p = var.test_whiteness(20)
        self.assertLessEqual(p, 0.01)       # now test should be significant


def main():
    unittest.main()


if __name__ == '__main__':
    main()
