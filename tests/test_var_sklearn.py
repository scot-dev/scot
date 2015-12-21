# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013-2014 SCoT Development Team

import unittest

import numpy as np
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoLars, ElasticNet

from scot.backend_sklearn import VAR


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

    def test_fit(self):
        np.random.seed(12345)
        x, var0 = self.generate_data()
        y = x.copy()

        var = VAR(2)
        var.fit(x)

        # make sure the input remains unchanged
        self.assertTrue(np.all(x == y))  # TODO: do we need this test?

        self.assertTrue(np.all(np.abs(var0.coef - var.coef) < 0.005))

    def test_residuals(self):
        np.random.seed(31415)
        x, var0 = self.generate_data()

        var = VAR(2)
        var.fit(x)

        self.assertEqual(x.shape, var.residuals.shape)
        self.assertTrue(np.allclose(var.rescov, np.eye(var.rescov.shape[0]),
                                    0.005, 0.005))

    def test_fit_ridge(self):
        b0, var = self._fit(Ridge(alpha=1))
        self.assertTrue(np.all(np.abs(b0 - var.coef) < 0.02))

    def test_fit_ridgecv(self):
        b0, var = self._fit(RidgeCV(alphas=np.logspace(-3, 3, 20)))
        self.assertTrue(np.all(np.abs(b0 - var.coef) < 0.02))

    def test_fit_lasso(self):
        b0, var = self._fit(Lasso(alpha=0.01))
        self.assertTrue(np.all(np.abs(b0 - var.coef) < 0.02))

    def test_fit_lassolars(self):
        b0, var = self._fit(LassoLars(alpha=0.00001))
        self.assertTrue(np.all(np.abs(b0 - var.coef) < 0.02))

    def test_fit_elasticnet(self):
        b0, var = self._fit(ElasticNet(alpha=0.01, l1_ratio=0.5))
        self.assertTrue(np.all(np.abs(b0 - var.coef) < 0.02))

    def _fit(self, estimator):
        np.random.seed(12345)
        x, var0 = self.generate_data()
        y = x.copy()

        var = VAR(10, estimator)
        var.fit(x)

        # make sure the input remains unchanged
        # self.assertTrue(np.all(x == y))  # TODO: do we need this test?

        b0 = np.zeros((2, 20))
        b0[:, 0:2] = var0.coef[:, 0:2]
        b0[:, 10:12] = var0.coef[:, 2:4]

        return b0, var
