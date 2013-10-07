import unittest
import sys

import numpy as np

from scot import datatools

class TestDataMangling(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass
    
    def test_cut_epochs(self):
        triggers = [100, 200, 300, 400, 500, 600, 700, 800, 900]
        rawdata = np.random.randn(1000,5)
        rawcopy = rawdata.copy()
        
        start = -10
        stop = 50
        
        X = datatools.cut_segments(rawdata, triggers, -10, 50)
        
        self.assertTrue(np.all(rawdata == rawcopy))
        self.assertEqual(X.shape, (stop-start, X.shape[1], len(triggers)))
        
        for it in range(len(triggers)):
            a = rawdata[triggers[it]+start : triggers[it]+stop, :]
            b = X[:,:,it]
            self.assertTrue(np.all(a == b))
            
    def test_cat_trials(self):
        X = np.random.randn(60,5,9)
        XC = X.copy()
        
        Y = datatools.cat_trials(X)
        
        self.assertTrue(np.all(X == XC))
        self.assertEqual(Y.shape, (X.shape[0]*X.shape[2], X.shape[1]))
        
        for it in range(X.shape[2]):
            a = Y[it*X.shape[0] : (it+1)*X.shape[0], :]
            b = X[:,:,it]
            self.assertTrue(np.all(a == b))
            
    def test_dot_special(self):
        X = np.random.randn(60,5,9)
        A = np.eye(5) * 2;
        
        XC = X.copy()
        AC = A.copy()
        
        Y = datatools.dot_special(X, A)
        
        self.assertTrue(np.all(X == XC))
        self.assertTrue(np.all(A == AC))        
        self.assertTrue(np.all(X*2 == Y))

class TestRegressions(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass
    
    def test_cat_trials_dimensions(self):
        """cat_trials did not always return a 2d array."""
        self.assertEqual(datatools.cat_trials(np.random.randn(100,2,2)).ndim, 2)
        self.assertEqual(datatools.cat_trials(np.random.randn(100,1,2)).ndim, 2)
        self.assertEqual(datatools.cat_trials(np.random.randn(100,2,1)).ndim, 2)
        self.assertEqual(datatools.cat_trials(np.random.randn(100,1,1)).ndim, 2)
        
def main():
    unittest.main()

if __name__ == '__main__':
    main()
