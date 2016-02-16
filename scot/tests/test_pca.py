# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013-2015 SCoT Development Team

import unittest

import numpy as np

from scot.pca import pca

try:
    from generate_testdata import generate_covsig
except ImportError:
    from .generate_testdata import generate_covsig

epsilon = 1e-10


class TestFunctionality(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testIdentity(self):
        """identity covariance in -> identity covariance out
           test for up to 50 dimensions
        """
        for i in range(1, 50):
            x = generate_covsig(np.eye(i), 500)
            w, v = pca(x)
            c = np.cov(np.dot(w.T, x))
            self.assertTrue(np.allclose(c, np.eye(i)))

    def testSorting(self):
        """components should be sorted by decreasing variance
        """
        x = generate_covsig(np.diag([1, 9, 2, 6, 3, 8, 4, 5, 7]), 500)
        w, v = pca(x, sort_components=True)
        c = np.cov(np.dot(w.T, x))
        self.assertTrue(np.allclose(c, np.diag([9, 8, 7, 6, 5, 4, 3, 2, 1]), rtol=1e-1, atol=1e-2))
        w, v = pca(x, sort_components=True)
        c = np.cov(np.dot(w.T, x))
        self.assertTrue(np.allclose(c, np.diag([9, 8, 7, 6, 5, 4, 3, 2, 1]), rtol=1e-1, atol=1e-2))

    def testDecorrelation(self):
        """components should be decorrelated after PCA
        """
        x = generate_covsig([[3, 2, 1], [2, 3, 2], [1, 2, 3]], 500)
        w, v = pca(x)
        c = np.cov(np.dot(w.T, x))
        c -= np.diag(c.diagonal())
        self.assertTrue(np.allclose(c, np.zeros((3, 3)), rtol=1e-2, atol=1e-3))


class TestDefaults(unittest.TestCase):
    def setUp(self):
        self.x = np.random.rand(10, 100)
        self.y = self.x.copy()
        self.m, self.n = self.x.shape
        self.w, self.v = pca(self.x)

    def tearDown(self):
        pass

    def testInputSafety(self):
        self.assertTrue((self.x == self.y).all())

        pca(self.x, subtract_mean=True, normalize=True)
        self.assertTrue((self.x == self.y).all())

    def testOutputSizes(self):
        self.assertTrue(self.w.shape == (self.m, self.m))
        self.assertTrue(self.v.shape == (self.m, self.m))

    def testInverse(self):
        i = np.abs(self.v.dot(self.w))
        self.assertTrue(np.abs(np.mean(i.diagonal())) - 1 < epsilon)
        self.assertTrue(np.abs(np.sum(i) - i.trace()) < epsilon)

        w, v = pca(self.x, subtract_mean=True, normalize=True)
        i = np.abs(v.dot(w))
        self.assertTrue(np.abs(np.mean(i.diagonal())) - 1 < epsilon)
        self.assertTrue(np.abs(np.sum(i) - i.trace()) < epsilon)


class TestDimensionalityReduction(unittest.TestCase):
    def setUp(self):
        self.x = np.random.rand(10, 100)
        self.y = self.x.copy()
        self.m, self.n = self.x.shape
        self.w1, self.v1 = pca(self.x, reducedim=0.9)
        self.w2, self.v2 = pca(self.x, reducedim=5)

    def tearDown(self):
        pass

    def testOutputSizes(self):
        self.assertTrue(self.w2.shape == (self.m, 5))
        self.assertTrue(self.v2.shape == (5, self.m))

    def testPseudoInverse(self):
        i = self.v1.dot(self.w1)
        self.assertTrue(np.abs(np.mean(i.diagonal()) - 1) < epsilon)

        i = self.w1.dot(self.v1)
        self.assertFalse(np.abs(np.mean(i.diagonal()) - 1) < epsilon)

        i = self.v2.dot(self.w2)
        self.assertTrue(np.abs(np.mean(i.diagonal()) - 1) < epsilon)

        i = self.w2.dot(self.v2)
        self.assertFalse(np.abs(np.mean(i.diagonal()) - 1) < epsilon)

    def testSorting(self):
        """components should be sorted by decreasing variance
        """
        x = generate_covsig(np.diag([1, 9, 2, 6, 3, 8, 4, 5, 7]), 500)
        w, v = pca(x, reducedim=5)
        c = np.cov(np.dot(w.T, x))
        self.assertTrue(np.allclose(c, np.diag([9, 8, 7, 6, 5]), rtol=1e-1, atol=1e-2))
