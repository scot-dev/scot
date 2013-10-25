# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

import unittest
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
        self.assertRaises(TypeError, scot.Workspace)
        api = scot.Workspace(var_order=50)
<<<<<<< HEAD
        self.assertRaises(RuntimeError, api.doMVARICA)
        self.assertRaises(RuntimeError, api.doCSPVARICA)
        self.assertRaises(RuntimeError, api.fitVAR)
        self.assertRaises(TypeError, api.getConnectivity)
        self.assertRaises(RuntimeError, api.getConnectivity, 'S')
        self.assertRaises(RuntimeError, api.getTFConnectivity, 'PDC', 10, 1)
=======
        self.assertRaises(RuntimeError, api.do_mvarica)
        self.assertRaises(RuntimeError, api.fit_var)
        self.assertRaises(TypeError, api.get_connectivity)
        self.assertRaises(RuntimeError, api.get_connectivity, 'S')
        self.assertRaises(RuntimeError, api.get_tf_connectivity, 'PDC', 10, 1)
>>>>>>> a80b5979f6814486393e66e7cfb9454d0c049aff
    
    def testModelIdentification(self):
        """ generate VAR signals, mix them, and see if MVARICA can reconstruct the signals
            do this for every backend """
        
        # original model coefficients
        b0 = np.zeros((3,6))
        b0[1:3,2:6] = [[ 0.4, -0.2, 0.3, 0.0],
                       [-0.7,  0.0, 0.9, 0.0]]            
        m0 = b0.shape[0]
        l, t = 1000, 100
        
        # generate VAR sources with non-gaussian innovation process, otherwise ICA won't work
        noisefunc = lambda: np.random.normal( size=(1,m0) )**3
        sources = var.simulate( [l,t], b0, noisefunc )
        
        # simulate volume conduction... 3 sources measured with 7 channels
        mix = [[0.5, 1.0, 0.5, 0.2, 0.0, 0.0, 0.0],
               [0.0, 0.2, 0.5, 1.0, 0.5, 0.2, 0.0],
               [0.0, 0.0, 0.0, 0.2, 0.5, 1.0, 0.5]]               
        data = datatools.dot_special(sources, mix)
        
        backend_modules = [import_module('scot.backend.'+b) for b in scot.backend.__all__]
        
        for bm in backend_modules:
            
            api = scot.Workspace(var_order=2, backend=bm.backend)
            
            api.set_data(data)
            
            # apply MVARICA
            #  - default setting of 0.99 variance should reduce to 3 channels with this data
            #  - automatically determine delta (enough data, so it should most likely be 0)
            api.do_mvarica()
            #result = varica.mvarica(data, 2, delta='auto', backend=bm.backend)
            
            # ICA does not define the ordering and sign of components
            # so wee need to test all combinations to find if one of them fits the original coefficients
            permutations = np.array([[0,1,2,3,4,5],[0,1,4,5,2,3],[2,3,4,5,0,1],[2,3,0,1,4,5],[4,5,0,1,2,3],[4,5,2,3,0,1]])
            signperms = np.array([[1,1,1,1,1,1], [1,1,1,1,-1,-1], [1,1,-1,-1,1,1], [1,1,-1,-1,-1,-1], [-1,-1,1,1,1,1], [-1,-1,1,1,-1,-1], [-1,-1,-1,-1,1,1], [-1,-1,-1,-1,-1,-1]])
            
            best, d = np.inf, None
    
            for perm in permutations:
                b = api.var_model_[perm[::2]//2,:]
                b = b[:,perm]
                for sgn in signperms:
                    c = b * np.repeat([sgn],3,0) * np.repeat([sgn[::2]],6,0).T
                    err = np.sum((c-b0)**2)
                    if err < best:
                        best = err
                        d = c
                        
            self.assertTrue(np.all(abs(d-b0) < 0.05))
    
    def testFunctionality(self):
        """ generate VAR signals, and apply the api to them
            do this for every backend """
        
        # original model coefficients
        b0 = np.zeros((3,6))
        b0[1:3,2:6] = [[ 0.4, -0.2, 0.3, 0.0],
                       [-0.7,  0.0, 0.9, 0.0]]            
        m0 = b0.shape[0]
        l, t = 1000, 10
        
        # generate VAR sources with non-gaussian innovation process, otherwise ICA won't work
        noisefunc = lambda: np.random.normal( size=(1,m0) )**3
        sources = var.simulate( [l,t], b0, noisefunc )
        
        # simulate volume conduction... 3 sources measured with 7 channels
        mix = [[0.5, 1.0, 0.5, 0.2, 0.0, 0.0, 0.0],
               [0.0, 0.2, 0.5, 1.0, 0.5, 0.2, 0.0],
               [0.0, 0.0, 0.0, 0.2, 0.5, 1.0, 0.5]]               
        data = datatools.dot_special(sources, mix)
        
        cl = [0,1,0,1,0,0,1,1,1,0]
        
        backend_modules = [import_module('scot.backend.'+b) for b in scot.backend.__all__]
        
        for bm in backend_modules:
            
            api = scot.Workspace(var_order=2, reducedim=3, backend=bm.backend)
            
            api.set_data(data)
            
            api.do_ica()
            
            self.assertEqual(api.mixing_.shape, (3,7))
            self.assertEqual(api.unmixing_.shape, (7,3))
            
            api.do_mvarica()
            
            self.assertEqual(api.get_connectivity('S').shape, (3,3,512))
            
            api.set_data(data)
            
            api.fit_var()
            
            self.assertEqual(api.get_connectivity('S').shape, (3,3,512))
            self.assertEqual(api.get_tf_connectivity('S', 100, 50).shape, (3,3,512,18))
            
            api.set_data(data, cl)
            
<<<<<<< HEAD
            api.doCSPVARICA()
            
            self.assertEqual(api.getConnectivity('S').shape, (3,3,512))
            
            api.fitVAR()
=======
            api.fit_var()
>>>>>>> a80b5979f6814486393e66e7cfb9454d0c049aff
                        
            fc = api.get_connectivity('S')
            tfc = api.get_tf_connectivity('S', 100, 50)
            for c in tfc:
                self.assertEqual(fc[c].shape, (3,3,512))
                self.assertEqual(tfc[c].shape, (3,3,512,18))
                            
            api.set_data(data)
            api.remove_sources([0,2])
            api.fit_var()
            self.assertEqual(api.get_connectivity('S').shape, (1,1,512))
            self.assertEqual(api.get_tf_connectivity('S', 100, 50).shape, (1,1,512,18))
            
            
                
        
        
def main():
    unittest.main()

if __name__ == '__main__':
    main()
