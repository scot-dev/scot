# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

import unittest
import sys

import numpy as np

from scot.builtin.pca import pca

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
        for i in range(1,50):
            X = generate_covsig(np.eye(i), 500)
            W, V = pca(X)
            C = np.cov(W.dot(X.T))
            self.assertTrue(np.allclose(C, np.eye(i)))
    
    def testSorting(self):
        """components should be sorted by decreasing variance
        """
        X = generate_covsig(np.diag([1,9,2,6,3,8,4,5,7]), 500)
        W, V = pca(X)
        C = np.cov(X.dot(W).T)
        self.assertTrue(np.allclose(C, np.diag([9,8,7,6,5,4,3,2,1]), rtol=1e-1, atol=1e-2))
        W, V = pca(X)
        C = np.cov(X.dot(W).T)
        self.assertTrue(np.allclose(C, np.diag([9,8,7,6,5,4,3,2,1]), rtol=1e-1, atol=1e-2))
    
    def testDecorrelation(self):
        """components should be decorrelated after PCA
        """
        X = generate_covsig([[3,2,1],[2,3,2],[1,2,3]], 500)
        W, V = pca(X)
        C = np.cov(X.dot(W).T)
        C -= np.diag(C.diagonal())
        self.assertTrue(np.allclose(C, np.zeros((3,3)), rtol=1e-2, atol=1e-3))

class TestDefaults(unittest.TestCase):

    def setUp(self):
        self.X = np.random.rand(100,10)
        self.Y = self.X.copy()
        self.N, self.M = self.X.shape
        self.W, self.V = pca(self.X)

    def tearDown(self):
        pass
    
    def testInputSafety(self):
        self.assertTrue((self.X == self.Y).all())
        
        W, V = pca(self.X, subtract_mean=True, normalize=True)
        self.assertTrue((self.X == self.Y).all())

    def testOutputSizes(self):
        self.assertTrue(self.W.shape == (self.M, self.M))
        self.assertTrue(self.V.shape == (self.M, self.M))

    def testInverse(self):
        I = np.abs(self.V.dot(self.W))        
        self.assertTrue(np.abs(np.mean(I.diagonal())) - 1 < epsilon)
        self.assertTrue(np.abs(np.sum(I) - I.trace()) < epsilon)        
        
        W, V = pca(self.X, subtract_mean=True, normalize=True)
        I = np.abs(V.dot(W))        
        self.assertTrue(np.abs(np.mean(I.diagonal())) - 1 < epsilon)
        self.assertTrue(np.abs(np.sum(I) - I.trace()) < epsilon)


class TestDimensionalityReduction(unittest.TestCase):

    def setUp(self):
        self.X = np.random.rand(100,10)
        self.Y = self.X.copy()
        self.N, self.M = self.X.shape
        self.W1, self.V1 = pca(self.X, retain_variance=0.9)
        self.W2, self.V2 = pca(self.X, numcomp=5)

    def tearDown(self):
        pass

    def testOutputSizes(self):
        self.assertTrue(self.W2.shape == (self.M, 5))
        self.assertTrue(self.V2.shape == (5, self.M))

    def testPseudoInverse(self):
        I = self.V1.dot(self.W1)        
        self.assertTrue(np.abs(np.mean(I.diagonal()) - 1) < epsilon)
        
        I = self.W1.dot(self.V1)        
        self.assertFalse(np.abs(np.mean(I.diagonal()) - 1) < epsilon)
        
        I = self.V2.dot(self.W2)        
        self.assertTrue(np.abs(np.mean(I.diagonal()) - 1) < epsilon)
        
        I = self.W2.dot(self.V2)
        self.assertFalse(np.abs(np.mean(I.diagonal()) - 1) < epsilon)
    
    def testSorting(self):
        """components should be sorted by decreasing variance
        """
        X = generate_covsig(np.diag([1,9,2,6,3,8,4,5,7]), 500)
        W, V = pca(X, numcomp=5)
        C = np.cov(X.dot(W).T)
        self.assertTrue(np.allclose(C, np.diag([9,8,7,6,5]), rtol=1e-1, atol=1e-2))
        
def main():
    unittest.main()

if __name__ == '__main__':
    main()
