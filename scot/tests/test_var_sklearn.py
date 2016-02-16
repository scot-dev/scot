# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013-2015 SCoT Development Team

import unittest

from numpy.testing.utils import assert_array_almost_equal
from numpy.testing.utils import assert_equal

import numpy as np
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoLars, ElasticNet

from scot.backend_sklearn import generate


backend_sklearn = generate()
VAR = backend_sklearn['var']


class CommonTests(unittest.TestCase):
    def setUp(self):
        np.random.seed(12345)
        self.var0 = VAR(2)
        self.var0.coef = np.array([[0.2, 0.1, 0.4, -0.1], [0.3, -0.2, 0.1, 0]])
        self.x = self.var0.simulate((1000, 100))
        self.var = VAR(2)

    def tearDown(self):
        pass

    def test_fit(self):
        self.var.fit(self.x)

        b0 = np.zeros_like(self.var.coef)
        b0[:, 0: 2] = self.var0.coef[:, 0:2]
        b0[:, self.var.p: self.var.p + 2] = self.var0.coef[:, 2: 4]

        assert_array_almost_equal(b0, self.var.coef, decimal=2)
        self.assertEqual(self.x.shape, self.var.residuals.shape)
        assert_array_almost_equal(self.var.rescov,
                                  np.eye(self.var.rescov.shape[0]), decimal=2)


class TestRidge(CommonTests):
    def setUp(self):
        super(TestRidge, self).setUp()
        self.var = VAR(10, Ridge(alpha=100))


class TestRidgeCV(CommonTests):
    def setUp(self):
        super(TestRidgeCV, self).setUp()
        # Provide three candidates for alpha.
        self.var = VAR(10, RidgeCV(alphas=[10, 100, 1000]))

    def test_alpha(self):
        """ This test checks if RidgeCV finds the optimal `alpha`.
        """
        self.var.fit(self.x)
        # Currently we simply *know* empirically that from the three
        # candidate alphas 100 is closest to the optimum.
        # TODO: programmatically derive the optimum from the data
        assert_equal(self.var.fitting_model.alpha_, 100)


class TestLasso(CommonTests):
    def setUp(self):
        super(TestLasso, self).setUp()
        self.var = VAR(10, Lasso(alpha=0.001))


class TestLassoLars(CommonTests):
    def setUp(self):
        super(TestLassoLars, self).setUp()
        self.var = VAR(10, LassoLars(alpha=0.00001))


class TestElasticNet(CommonTests):
    def setUp(self):
        super(TestElasticNet, self).setUp()
        self.var = VAR(10, ElasticNet(alpha=0.01, l1_ratio=0.5))
