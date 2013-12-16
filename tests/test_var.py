# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

import unittest

import numpy as np

from scot import var

epsilon = 1e-10


class TestVAR(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_simulate(self):
        noisefunc = lambda: [1, 1]   # use deterministic function instead of noise
        b = np.array([[0.2, 0.1, 0.4, -0.1], [0.3, -0.2, 0.1, 0]])
        num_samples = 100
        x = var.simulate(num_samples, b, noisefunc)

        # make sure we got expected values within reasonable accuracy
        for n in range(10, num_samples):
            self.assertTrue(np.all(
                np.abs(x[n, :] - 1 - np.dot(b[:, 0::2], x[n - 1, :]) - np.dot(b[:, 1::2], x[n - 2, :])) < epsilon))

    def test_fit(self):
        b = np.array([[0.2, 0.1, 0.4, -0.1], [0.3, -0.2, 0.1, 0]])
        l = 100000
        x = var.simulate(l, b)
        y = x.copy()

        b_fit = var.fit(x, 2)

        # make sure the input remains unchanged
        self.assertTrue(np.all(x == y))

        # that limit is rather generous, but we don't want tests to fail due to random variation
        self.assertTrue(np.all(np.abs(b - b_fit) < 0.02))

    def test_fit_regularized(self):
        b = np.array([[0.2, 0.1, 0.4, -0.1], [0.3, -0.2, 0.1, 0]])
        l = 100000
        x = var.simulate(l, b)
        y = x.copy()

        b_fit = var.fit(x, 10, delta=1)

        # make sure the input remains unchanged
        self.assertTrue(np.all(x == y))

        b0 = np.zeros((2, 20))
        b0[:, 0:2] = b[:, 0:2]
        b0[:, 10:12] = b[:, 2:4]

        # that limit is rather generous, but we don't want tests to fail due to random variation
        self.assertTrue(np.all(np.abs(b0 - b_fit) < 0.02))

    def test_predict(self):
        b = np.array([[0.2, 0.1, 0.4, -0.1], [0.3, -0.2, 0.1, 0]])
        l = 100000
        x = var.simulate(l, b)

        z = var.predict(x, b)

        # that limit is rather generous, but we don't want tests to fail due to random variation
        self.assertTrue(np.abs(np.var(x[100:, :] - z[100:, :]) - 1) < 0.02)

    def test_residuals(self):
        b = np.array([[0.2, 0.1, 0.4, -0.1], [0.3, -0.2, 0.1, 0]])
        l = 100000
        x = var.simulate(l, b)

        _, r, c = var.fit(x, 2, return_residuals=True, return_covariance=True)

        self.assertEqual(x.shape, r.shape)

        self.assertTrue(np.allclose(c, np.eye(c.shape[0]), 1e-2, 1e-2))

    def test_whiteness(self):
        r = np.random.randn(100,5,10)
        r0 = r.copy()
        p = var.test_whiteness(r, 0, 20)

        self.assertTrue(np.all(r == r0))
        self.assertTrue(p>0.05)

        r[3:,1,:] = r[:-3,0,:]
        p = var.test_whiteness(r, 0, 20)
        self.assertFalse(p>0.05)



def main():
    unittest.main()


if __name__ == '__main__':
    main()
