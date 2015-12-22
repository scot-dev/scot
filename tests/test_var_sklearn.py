# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013-2014 SCoT Development Team

import unittest

import numpy as np
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoLars, ElasticNet

from scot.backend_sklearn import VAR


class TestVAR(unittest.TestCase):
    def setUp(self):
        np.random.seed(12345)
        self.var0 = VAR(2)
        self.var0.coef = np.array([[0.2, 0.1, 0.4, -0.1], [0.3, -0.2, 0.1, 0]])
        self.x = self.var0.simulate((1000, 100))

    def tearDown(self):
        pass

    def test_fit(self):
        var = VAR(2)
        var.fit(self.x)
        self.assertTrue(np.all(np.abs(self.var0.coef - var.coef) < 0.005))

    def test_residuals(self):
        var = VAR(2)
        var.fit(self.x)

        self.assertEqual(self.x.shape, var.residuals.shape)
        self.assertTrue(np.allclose(var.rescov,
                                    np.eye(var.rescov.shape[0]), 0.005, 0.005))

    def test_fit_ridge(self):
        var = VAR(10, Ridge(alpha=1))
        var.fit(self.x)
        b0 = np.zeros((2, 20))
        b0[:, 0:2] = self.var0.coef[:, 0:2]
        b0[:, 10:12] = self.var0.coef[:, 2:4]
        self.assertTrue(np.all(np.abs(b0 - var.coef) < 0.02))

    def test_fit_ridgecv(self):
        var = VAR(10, RidgeCV(alphas=np.logspace(-3, 3, 20)))
        var.fit(self.x)
        b0 = np.zeros((2, 20))
        b0[:, 0:2] = self.var0.coef[:, 0:2]
        b0[:, 10:12] = self.var0.coef[:, 2:4]
        self.assertTrue(np.all(np.abs(b0 - var.coef) < 0.02))

    def test_fit_lasso(self):
        var = VAR(10, Lasso(alpha=0.01))
        var.fit(self.x)
        b0 = np.zeros((2, 20))
        b0[:, 0:2] = self.var0.coef[:, 0:2]
        b0[:, 10:12] = self.var0.coef[:, 2:4]
        self.assertTrue(np.all(np.abs(b0 - var.coef) < 0.02))

    def test_fit_lassolars(self):
        var = VAR(10, LassoLars(alpha=0.00001))
        var.fit(self.x)
        b0 = np.zeros((2, 20))
        b0[:, 0:2] = self.var0.coef[:, 0:2]
        b0[:, 10:12] = self.var0.coef[:, 2:4]
        self.assertTrue(np.all(np.abs(b0 - var.coef) < 0.02))

    def test_fit_elasticnet(self):
        var = VAR(10, ElasticNet(alpha=0.01, l1_ratio=0.5))
        var.fit(self.x)
        b0 = np.zeros((2, 20))
        b0[:, 0:2] = self.var0.coef[:, 0:2]
        b0[:, 10:12] = self.var0.coef[:, 2:4]
        self.assertTrue(np.all(np.abs(b0 - var.coef) < 0.02))
