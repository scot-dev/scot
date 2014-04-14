# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

import unittest

import numpy as np

from scot.builtin.var import VAR

epsilon = 1e-10


class TestVAR(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_simulate(self):
        noisefunc = lambda: [1, 1]   # use deterministic function instead of noise
        num_samples = 100

        b = np.array([[0.2, 0.1, 0.4, -0.1], [0.3, -0.2, 0.1, 0]])

        var = VAR(2)
        var.coef = b

        x = var.simulate(num_samples, noisefunc)

        # make sure we got expected values within reasonable accuracy
        for n in range(10, num_samples):
            self.assertTrue(np.all(
                np.abs(x[n, :] - 1 - np.dot(b[:, 0::2], x[n - 1, :]) - np.dot(b[:, 1::2], x[n - 2, :])) < epsilon))

    def test_fit(self):
        var0 = VAR(2)
        var0.coef = np.array([[0.2, 0.1, 0.4, -0.1], [0.3, -0.2, 0.1, 0]])
        l = 100000
        x = var0.simulate(l)
        y = x.copy()

        var = VAR(2)
        var.fit(x)

        # make sure the input remains unchanged
        self.assertTrue(np.all(x == y))

        # that limit is rather generous, but we don't want tests to fail due to random variation
        self.assertTrue(np.all(np.abs(var0.coef - var.coef) < 0.02))

    def test_fit_regularized(self):
        l = 100000
        var0 = VAR(2)
        var0.coef = np.array([[0.2, 0.1, 0.4, -0.1], [0.3, -0.2, 0.1, 0]])
        x = var0.simulate(l)
        y = x.copy()

        var = VAR(10, delta=1)
        var.fit(x)

        # make sure the input remains unchanged
        self.assertTrue(np.all(x == y))

        b0 = np.zeros((2, 20))
        b0[:, 0:2] = var0.coef[:, 0:2]
        b0[:, 10:12] = var0.coef[:, 2:4]

        # that limit is rather generous, but we don't want tests to fail due to random variation
        self.assertTrue(np.all(np.abs(b0 - var.coef) < 0.02))

    def test_predict(self):
        l = 100000
        var = VAR(2)
        var.coef = np.array([[0.2, 0.1, 0.4, -0.1], [0.3, -0.2, 0.1, 0]])

        x = var.simulate(l)
        z = var.predict(x)

        # that limit is rather generous, but we don't want tests to fail due to random variation
        self.assertTrue(np.abs(np.var(x[100:, :] - z[100:, :]) - 1) < 0.02)

    def test_residuals(self):
        l = 100000
        var0 = VAR(2)
        var0.coef = np.array([[0.2, 0.1, 0.4, -0.1], [0.3, -0.2, 0.1, 0]])
        x = var0.simulate(l)

        var = VAR(2)
        var.fit(x)

        self.assertEqual(x.shape, var.residuals.shape)

        self.assertTrue(np.allclose(var.rescov, np.eye(var.rescov.shape[0]), 1e-2, 1e-2))

    def test_whiteness(self):
        r = np.random.randn(100, 5, 10)     # gaussian white noise
        r0 = r.copy()

        var = VAR(0)
        var.residuals = r

        p = var.test_whiteness(20)

        self.assertTrue(np.all(r == r0))    # make sure we don't modify the input
        self.assertTrue(p>0.05)             # test should be non-significant for white noise

        r[3:,1,:] = r[:-3,0,:]              # create cross-correlation at lag 3
        p = var.test_whiteness(20)
        self.assertFalse(p>0.05)            # now test should be significant



def main():
    unittest.main()


if __name__ == '__main__':
    main()
