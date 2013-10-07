import unittest
import sys

import numpy as np

import scot.backend.builtin
from scot import varica, var, datatools

class TestMVARICA(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    def testModelIdentification(self):
        """ generate VAR signals, mix them, and see if MVARICA can reconstruct the signals """
        
        scot.backend.builtin.activate()     # this line is only necessary when changing the backend multiple times in one interpreter session
        
        # original model coefficients
        B0 = np.zeros((3,6))
        B0[1:3,2:6] = [[ 0.4, -0.2, 0.3, 0.0],
                       [-0.7,  0.0, 0.9, 0.0]]            
        M0 = B0.shape[0]
        L, T = 1000, 100
        
        # generate VAR sources with non-gaussian innovation process, otherwise ICA won't work
        noisefunc = lambda: np.random.normal( size=(1,M0) )**3   
        sources = var.simulate( [L,T], B0, noisefunc )
        
        # simulate volume conduction... 3 sources measured with 7 channels
        mix = [[0.5, 1.0, 0.5, 0.2, 0.0, 0.0, 0.0],
               [0.0, 0.2, 0.5, 1.0, 0.5, 0.2, 0.0],
               [0.0, 0.0, 0.0, 0.2, 0.5, 1.0, 0.5]]               
        data = datatools.dot_special(sources, mix)
        
        # apply MVARICA
        #  - default setting of 0.99 variance should reduce to 3 channels with this data
        #  - automatically determine delta (enough data, so it should most likely be 0)
        result = varica.mvarica(data, 2, delta='auto')
        
        # ICA does not define the ordering and sign of components
        # so wee need to test all combinations to find if one of them fits the original coefficients
        permutations = np.array([[0,1,2,3,4,5],[0,1,4,5,2,3],[2,3,4,5,0,1],[2,3,0,1,4,5],[4,5,0,1,2,3],[4,5,2,3,0,1]])
        signperms = np.array([[1,1,1,1,1,1], [1,1,1,1,-1,-1], [1,1,-1,-1,1,1], [1,1,-1,-1,-1,-1], [-1,-1,1,1,1,1], [-1,-1,1,1,-1,-1], [-1,-1,-1,-1,1,1], [-1,-1,-1,-1,-1,-1]])
        
        best = np.inf

        for perm in permutations:
            B = result.B[perm[::2]//2,:]
            B = B[:,perm]
            for sgn in signperms:
                C = B * np.repeat([sgn],3,0) * np.repeat([sgn[::2]],6,0).T        
                d = np.sum((C-B0)**2)
                if d < best:
                    best = d
                    D = C
                    
        print(D-B0)
        self.assertTrue(np.all(abs(D-B0) < 0.05))
                
        
        
def main():
    unittest.main()

if __name__ == '__main__':
    main()
