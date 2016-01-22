# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013-2015 SCoT Development Team

import unittest

import numpy as np
from numpy.testing.utils import assert_allclose

from scot.csp import csp

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
    
    def testComponentSeparation(self):
        A = generate_covsig([[10,5,2],[5,10,2],[2,2,10]], 500)
        B = generate_covsig([[10,2,2],[2,10,5],[2,5,10]], 500)
            
        X = np.concatenate([A[np.newaxis], B[np.newaxis]], axis=0)
        W, V = csp(X, [1, 2])
        C1a = np.cov(np.dot(W.T, X[0, :, :]))
        C2a = np.cov(np.dot(W.T, X[1, :, :]))
        
        Y = np.concatenate([B[np.newaxis], A[np.newaxis]], axis=0)
        W, V = csp(Y, [1, 2])
        C1b = np.cov(np.dot(W.T, Y[0, :, :]))
        C2b = np.cov(np.dot(W.T, Y[1, :, :]))
        
        # check symmetric case
        assert_allclose(C1a.diagonal(), C2a.diagonal()[::-1])
        assert_allclose(C1b.diagonal(), C2b.diagonal()[::-1])
        
        # swapping class labels (or in this case, trials) should not change the result
        assert_allclose(C1a, C1b, rtol=1e-9, atol=1e-9)
        assert_allclose(C2a, C2b, rtol=1e-9, atol=1e-9)
        
        # variance of first component should be greatest for class 1
        self.assertGreater(C1a[0, 0], C2a[0, 0])
        
        # variance of last component should be greatest for class 1
        self.assertLess(C1a[2, 2], C2a[2, 2])
        
        # variance of central component should be equal for both classes
        assert_allclose(C1a[1, 1], C2a[1, 1])
        

class TestDefaults(unittest.TestCase):

    def setUp(self):
        self.X = np.random.randn(10,5,100)
        self.C = [0,0,0,0,0,1,1,1,1,1]
        self.Y = self.X.copy()
        self.D = list(self.C)
        self.T, self.M, self.N = self.X.shape
        self.W, self.V = csp(self.X, self.C)

    def tearDown(self):
        pass
    
    def testInvalidInput(self):
        # pass only 2d data
        self.assertRaises(AttributeError, csp, np.random.randn(3,10), [1,1,0,0] )
        
        # number of class labels does not match number of trials
        self.assertRaises(AttributeError, csp, np.random.randn(5,3,10), [1,1,0,0] )
    
    def testInputSafety(self):
        # function must not change input variables
        self.assertTrue((self.X == self.Y).all())        
        self.assertEqual(self.C, self.D)

    def testOutputSizes(self):
        # output matrices must have the correct size
        self.assertTrue(self.W.shape == (self.M, self.M))
        self.assertTrue(self.V.shape == (self.M, self.M))

    def testInverse(self):
        # V should be the inverse of W
        I = np.abs(self.V.dot(self.W))
        
        self.assertTrue(np.abs(np.mean(I.diagonal())) - 1 < epsilon)
        self.assertTrue(np.abs(np.sum(I) - I.trace()) < epsilon)


class TestDimensionalityReduction(unittest.TestCase):

    def setUp(self):
        self.n_comps = 5
        self.X = np.random.rand(10,15,100)
        self.C = [0,0,0,0,0,1,1,1,1,1]
        self.Y = self.X.copy()
        self.D = list(self.C)
        self.T, self.M, self.N = self.X.shape
        self.W, self.V = csp(self.X, self.C, numcomp=self.n_comps)

    def tearDown(self):
        pass

    def testOutputSizes(self):
        # output matrices must have the correct size
        self.assertTrue(self.W.shape == (self.M, 5))
        self.assertTrue(self.V.shape == (5, self.M))

    def testPseudoInverse(self):
        # V should be the pseudo inverse of W
        I = self.V.dot(self.W)
        assert_allclose(I, np.eye(self.n_comps), rtol=1e-9, atol=1e-9)
        
def main():
    unittest.main()

if __name__ == '__main__':
    main()
