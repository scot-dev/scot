# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013-2015 SCoT Development Team

import unittest

import numpy as np

from scot.var import VAR

epsilon = 1e-10


class TestVAR(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

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

    def test_residuals(self):
        l = 100000
        var0 = VAR(2)
        var0.coef = np.array([[0.2, 0.1, 0.4, -0.1], [0.3, -0.2, 0.1, 0]])
        x = var0.simulate(l)

        var = VAR(2)
        var.fit(x)

        self.assertEqual(x.shape, var.residuals.shape)

        self.assertTrue(np.allclose(var.rescov, np.eye(var.rescov.shape[0]), 1e-2, 1e-2))

    def test_optimize(self):
        np.random.seed(745)
        var0 = VAR(2)
        var0.coef = np.array([[0.2, 0.1, 0.4, -0.1], [0.3, -0.2, 0.1, 0]])
        l = (100, 10)
        x = var0.simulate(l)

        var = VAR(-1)
        for n_jobs in [None, -1, 1, 2]:
            var.optimize_order(x, verbose=True, n_jobs=n_jobs)
            self.assertEqual(var.p, 2)
