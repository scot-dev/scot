# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

import unittest
import sys
from importlib import import_module
import numpy as np

import scot.backend
from scot import var, datatools
import scot

class TestMVARICA(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    def testExceptions(self):        
        self.assertRaises(TypeError, scot.SCoT)
        api = scot.SCoT(var_order=50)
        self.assertRaises(RuntimeError, api.doMVARICA)
        self.assertRaises(RuntimeError, api.fitVAR)
        self.assertRaises(TypeError, api.getConnectivity)
        self.assertRaises(RuntimeError, api.getConnectivity, 'S')
        self.assertRaises(RuntimeError, api.getTFConnectivity, 'PDC', 10, 1)
    
#    def testModelIdentification(self):
#        """ generate VAR signals, mix them, and see if MVARICA can reconstruct the signals """
#        """ do this for every backend """
#        
#        # original model coefficients
#        B0 = np.zeros((3,6))
#        B0[1:3,2:6] = [[ 0.4, -0.2, 0.3, 0.0],
#                       [-0.7,  0.0, 0.9, 0.0]]            
#        M0 = B0.shape[0]
#        L, T = 1000, 100
#        
#        # generate VAR sources with non-gaussian innovation process, otherwise ICA won't work
#        noisefunc = lambda: np.random.normal( size=(1,M0) )**3   
#        sources = var.simulate( [L,T], B0, noisefunc )
#        
#        # simulate volume conduction... 3 sources measured with 7 channels
#        mix = [[0.5, 1.0, 0.5, 0.2, 0.0, 0.0, 0.0],
#               [0.0, 0.2, 0.5, 1.0, 0.5, 0.2, 0.0],
#               [0.0, 0.0, 0.0, 0.2, 0.5, 1.0, 0.5]]               
#        data = datatools.dot_special(sources, mix)
#        
#        backend_modules = [import_module('scot.backend.'+b) for b in scot.backend.__all__]
#        
#        for bm in backend_modules:
#            
#            api = scot.SCoT(var_order=2, backend=bm.backend)
#            
#            api.setData(data)
#            
#            # apply MVARICA
#            #  - default setting of 0.99 variance should reduce to 3 channels with this data
#            #  - automatically determine delta (enough data, so it should most likely be 0)
#            api.doMVARICA()
#            #result = varica.mvarica(data, 2, delta='auto', backend=bm.backend)
#            
#            # ICA does not define the ordering and sign of components
#            # so wee need to test all combinations to find if one of them fits the original coefficients
#            permutations = np.array([[0,1,2,3,4,5],[0,1,4,5,2,3],[2,3,4,5,0,1],[2,3,0,1,4,5],[4,5,0,1,2,3],[4,5,2,3,0,1]])
#            signperms = np.array([[1,1,1,1,1,1], [1,1,1,1,-1,-1], [1,1,-1,-1,1,1], [1,1,-1,-1,-1,-1], [-1,-1,1,1,1,1], [-1,-1,1,1,-1,-1], [-1,-1,-1,-1,1,1], [-1,-1,-1,-1,-1,-1]])
#            
#            best = np.inf
#    
#            for perm in permutations:
#                B = api.var_model_[perm[::2]//2,:]
#                B = B[:,perm]
#                for sgn in signperms:
#                    C = B * np.repeat([sgn],3,0) * np.repeat([sgn[::2]],6,0).T        
#                    d = np.sum((C-B0)**2)
#                    if d < best:
#                        best = d
#                        D = C
#                        
#            self.assertTrue(np.all(abs(D-B0) < 0.05))
    
    def testFunctionality(self):
        """ generate VAR signals, and apply the api to them """
        """ do this for every backend """
        
        # original model coefficients
        B0 = np.zeros((3,6))
        B0[1:3,2:6] = [[ 0.4, -0.2, 0.3, 0.0],
                       [-0.7,  0.0, 0.9, 0.0]]            
        M0 = B0.shape[0]
        L, T = 1000, 10
        
        # generate VAR sources with non-gaussian innovation process, otherwise ICA won't work
        noisefunc = lambda: np.random.normal( size=(1,M0) )**3   
        sources = var.simulate( [L,T], B0, noisefunc )
        
        # simulate volume conduction... 3 sources measured with 7 channels
        mix = [[0.5, 1.0, 0.5, 0.2, 0.0, 0.0, 0.0],
               [0.0, 0.2, 0.5, 1.0, 0.5, 0.2, 0.0],
               [0.0, 0.0, 0.0, 0.2, 0.5, 1.0, 0.5]]               
        data = datatools.dot_special(sources, mix)
        
        cl = [0,1,0,1,0,0,1,1,1,0]
        
        backend_modules = [import_module('scot.backend.'+b) for b in scot.backend.__all__]
        
        for bm in backend_modules:
            
            api = scot.SCoT(var_order=2, reducedim=3, backend=bm.backend)
            
            api.setData(data)
            
            api.doICA()
            
            self.assertEqual(api.mixing_.shape, (3,7))
            self.assertEqual(api.unmixing_.shape, (7,3))
            
            api.doMVARICA()
            
            self.assertEqual(api.getConnectivity('S').shape, (3,3,512))
            
            api.setData(data)
            
            api.fitVAR()
            
            self.assertEqual(api.getConnectivity('S').shape, (3,3,512))
            self.assertEqual(api.getTFConnectivity('S', 100, 50).shape, (3,3,18,512))
            
            api.setData(data, cl)
            
            api.fitVAR()
                        
            fc = api.getConnectivity('S')
            tfc = api.getTFConnectivity('S', 100, 50)
            for c in tfc:
                self.assertEqual(fc[c].shape, (3,3,512))
                self.assertEqual(tfc[c].shape, (3,3,18,512))
                            
            api.setData(data)
            api.removeSources([0,2])
            api.fitVAR()            
            self.assertEqual(api.getConnectivity('S').shape, (1,1,512))
            self.assertEqual(api.getTFConnectivity('S', 100, 50).shape, (1,1,18,512))
            
            
                
        
        
def main():
    unittest.main()

if __name__ == '__main__':
    main()
