# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

import unittest

import numpy as np
from sklearn import linear_model

from scot.backend.sklearn import VAR


class TestVAR(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def generate_data(self):
        var = VAR(2)
        var.coef = np.array([[0.2, 0.1, 0.4, -0.1], [0.3, -0.2, 0.1, 0]])
        l = (1000, 100)
        x = var.simulate(l)
        return x, var

    def test_simulate(self):
        noisefunc = lambda: [1, 1]   # use deterministic function instead of noise
        num_samples = 100

        b = np.array([[0.2, 0.1, 0.4, -0.1], [0.3, -0.2, 0.1, 0]])

        var = VAR(2)
        var.coef = b

        np.random.seed(42)
        x = var.simulate(num_samples, noisefunc)

        # make sure we got expected values within reasonable accuracy
        for n in range(10, num_samples):
            self.assertTrue(np.all(
                np.abs(x[n, :] - 1 - np.dot(b[:, 0::2], x[n - 1, :]) - np.dot(b[:, 1::2], x[n - 2, :])) < 1e-10))

    def test_fit(self):
        np.random.seed(12345)
        x, var0 = self.generate_data()
        y = x.copy()

        var = VAR(2)
        var.fit(x)

        # make sure the input remains unchanged
        self.assertTrue(np.all(x == y))

        self.assertTrue(np.all(np.abs(var0.coef - var.coef) < 0.005))

    def test_predict(self):
        np.random.seed(777)
        x, var = self.generate_data()
        z = var.predict(x)

        self.assertTrue(np.abs(np.var(x[100:, :] - z[100:, :]) - 1) < 0.005)

    def test_residuals(self):
        np.random.seed(31415)
        x, var0 = self.generate_data()

        var = VAR(2)
        var.fit(x)

        self.assertEqual(x.shape, var.residuals.shape)
        self.assertTrue(np.allclose(var.rescov, np.eye(var.rescov.shape[0]), 0.005, 0.005))

    def test_whiteness(self):
        np.random.seed(91)
        r = np.random.randn(100, 5, 10)     # gaussian white noise
        r0 = r.copy()

        var = VAR(0)
        var.residuals = r

        p = var.test_whiteness(20)

        self.assertTrue(np.all(r == r0))    # make sure we don't modify the input
        self.assertGreater(p, 0.01)         # test should be non-significant for white noise

        r[3:,1,:] = r[:-3,0,:]              # create cross-correlation at lag 3
        p = var.test_whiteness(20)
        self.assertLessEqual(p, 0.01)       # now test should be significant


# dynamically create testing functions for different fitting models
def create_func(o):
    def func(self):
        x, var0 = self.generate_data()
        y = x.copy()

        var = VAR(10, o)
        var.fit(x)

        # make sure the input remains unchanged
        self.assertTrue(np.all(x == y))

        b0 = np.zeros((2, 20))
        b0[:, 0:2] = var0.coef[:, 0:2]
        b0[:, 10:12] = var0.coef[:, 2:4]

        # that limit is rather generous, but we don't want tests to fail due to random variation
        self.assertTrue(np.all(np.abs(b0 - var.coef) < 0.02))
    return func

fmo = {'test_fit_Ridge': linear_model.Ridge(alpha=1),
       'test_fit_RidgeCV': linear_model.RidgeCV(alphas=np.logspace(-3, 3, 20)),
       'test_fit_Lasso': linear_model.Lasso(alpha=0.01),
       #'test_fit_LassoCV': linear_model.LassoCV(),
       'test_fit_ElasticNet': linear_model.ElasticNet(alpha=0.01, l1_ratio=0.5),
       #'test_fit_ElasticNetCV': linear_model.ElasticNetCV(),
       'test_fit_LassoLars': linear_model.LassoLars(alpha=0.00001),
       }

for f, o in fmo.items():
    setattr(TestVAR, f, create_func(o))


def main():
    unittest.main()


if __name__ == '__main__':
    main()
