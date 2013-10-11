# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

import unittest
import sys
from importlib import import_module
import numpy as np

import scot.backend
from scot import plainica, var, datatools

class TestICA(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    def testModelIdentification(self):
        """ generate independent signals, mix them, and see if ICA can reconstruct the mixing matrix """
        """ do this for every backend """
        
        # original model coefficients
        B0 = np.zeros((3,3))    # no connectivity
        M0 = B0.shape[0]
        L, T = 100, 100
        
        # generate VAR sources with non-gaussian innovation process, otherwise ICA won't work
        noisefunc = lambda: np.random.normal( size=(1,M0) )**3   
        sources = var.simulate( [L,T], B0, noisefunc )
        
        # simulate volume conduction... 3 sources measured with 7 channels
        mix = [[0.5, 1.0, 0.5, 0.2, 0.0, 0.0, 0.0],
               [0.0, 0.2, 0.5, 1.0, 0.5, 0.2, 0.0],
               [0.0, 0.0, 0.0, 0.2, 0.5, 1.0, 0.5]]               
        data = datatools.dot_special(sources, mix)
        
        backend_modules = [import_module('scot.backend.'+b) for b in scot.backend.__all__]
        
        for bm in backend_modules:
            
            result = plainica.plainica(data, backend=bm.backend)
            
            I = result.mixing.dot(result.unmixing)
            self.assertTrue(np.allclose(I,np.eye(I.shape[0]), rtol=1e-6, atol=1e-7))
            
            permutations = [[0,1,2], [0,2,1], [1,0,2], [1,2,0], [2,0,1], [2,1,0]]
            
            bestdiff = np.inf
            bestmix = None
            
            absmix = np.abs(result.mixing)
            absmix /= np.max(absmix)
            
            for p in permutations:
                estmix = absmix[p,:]
                diff = np.sum((np.abs(estmix)-np.abs(mix))**2)
                
                if diff < bestdiff:
                    bestdiff = diff
                    bestmix = estmix
            
            self.assertTrue(np.allclose(bestmix,mix, rtol=1e-1, atol=1e-1))
                        
            
                
        
        
def main():
    unittest.main()

if __name__ == '__main__':
    main()
