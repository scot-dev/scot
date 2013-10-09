# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

import unittest
import sys

import numpy as np

from scot import var

epsilon = 1e-10

class TestVAR(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass
    
#    def test_simulate(self):
#        noisefunc = lambda: [1,1]   # use deterministic function instead of noise
#        B = np.array([[0.2, 0.1, 0.4, -0.1], [0.3, -0.2, 0.1, 0]])
#        L = 100
#        X = var.simulate(L, B, noisefunc)
#        
#        # make sure we got expected values within reasonable accuracy
#        for n in range(10,L):
#            self.assertTrue( np.all(np.abs(X[n,:] - 1 - np.dot(B[:,0::2], X[n-1,:]) - np.dot(B[:,1::2], X[n-2,:])) < epsilon) )
#    
#    def test_fit(self):
#        B = np.array([[0.2, 0.1, 0.4, -0.1], [0.3, -0.2, 0.1, 0]])
#        L = 100000
#        X = var.simulate(L, B)
#        Y = X.copy()
#        
#        B_fit = var.fit(X, 2)
#
#        # make sure the input remains unchanged
#        self.assertTrue(np.all(X == Y))        
#        
#        # that limit is rather generous, but we don't want tests to fail due to random variation
#        self.assertTrue(np.all(np.abs(B-B_fit) < 0.02))
#    
#    def test_fit_regularized(self):
#        B = np.array([[0.2, 0.1, 0.4, -0.1], [0.3, -0.2, 0.1, 0]])
#        L = 100000
#        X = var.simulate(L, B)
#        Y = X.copy()
#        
#        B_fit = var.fit(X, 10, delta=1)
#
#        # make sure the input remains unchanged
#        self.assertTrue(np.all(X == Y))
#        
#        B0 = np.zeros((2,20))
#        B0[:,0:2] = B[:,0:2]
#        B0[:,10:12] = B[:,2:4]
#        
#        # that limit is rather generous, but we don't want tests to fail due to random variation
#        self.assertTrue(np.all(np.abs(B0-B_fit) < 0.02))
#    
#    def test_predict(self):
#        B = np.array([[0.2, 0.1, 0.4, -0.1], [0.3, -0.2, 0.1, 0]])
#        L = 100000
#        X = var.simulate(L, B)
#        Y = X.copy()
#        
#        Z = var.predict(X, B)
#        
#        # that limit is rather generous, but we don't want tests to fail due to random variation
#        self.assertTrue( np.abs(np.var(X[100:,:]-Z[100:,:]) - 1) < 0.02 )
        
    def test_residuals(self):
        B = np.array([[0.2, 0.1, 0.4, -0.1], [0.3, -0.2, 0.1, 0]])
        L = 100000
        X = var.simulate(L, B)
        
        _, R, C = var.fit(X, 2, return_residuals=True, return_covariance=True)
        
        self.assertEqual(X.shape, R.shape)
        
        self.assertTrue(np.allclose(C, np.eye(C.shape[0]), 1e-2, 1e-2))
        
def main():
    unittest.main()

if __name__ == '__main__':
    main()
