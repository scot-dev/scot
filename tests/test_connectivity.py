# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

import unittest
import sys

import numpy as np

from scot import connectivity

class TestFunctionality(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass
    
    def testSimple(self):
        # Three sources: A <- B <-C
        # simply test if connectivity measures are 0 where expected to be so
        B0 = np.array([[0,0.9,0],[0,0,0.9],[0,0,0]])
        I = np.eye(3)
        Nfft = 5
        c = connectivity.Connectivity(B=B0, C=I, Nfft=Nfft)
        K = lambda x: np.sum(np.abs(x),2)
        L = lambda x: np.sum(x,2)
        # A should have the same structure as B
        self.assertTrue(np.all((K(c.A())==0) == ((B0+I)==0)))
        self.assertFalse(np.all(K(c.A()) == K(c.A()).T))
        # H should be upper triangular
        self.assertTrue(np.all(np.tril(K(c.H()), -1) == 0))
        self.assertFalse(np.all(K(c.H()) == K(c.H()).T))
        # S should be a full matrix and symmetric
        self.assertTrue(np.all(K(c.S()) > 0 ))
        self.assertTrue(np.all(K(c.S()) == K(c.S()).T))
        # G should be nonzero for direct connections only and symmetric in magnitude
        self.assertEqual(K(c.G())[0,2], 0)
        self.assertTrue(np.all(K(c.G()) == K(c.G()).T))
        # Phase should be zero along the diagonal
        self.assertTrue(np.all(K(c.PHI()).diagonal() == 0))
        # Phase should be zero along the diagonal and antisymmetric
        self.assertTrue(np.all(K(c.PHI()).diagonal() == 0))
        self.assertTrue(np.all(L(c.PHI()) == -L(c.PHI()).T))
        # Coherence should be 1 over all frequencies along the diagonal
        self.assertTrue(np.all(K(c.COH()).diagonal() == Nfft))
        # pCOH should be nonzero for direct connections only and symmetric in magnitude
        self.assertEqual(K(c.pCOH())[0,2], 0)
        self.assertTrue(np.all(K(c.pCOH()) == K(c.pCOH()).T))
        # PDC should have the same structure as B,
        self.assertTrue(np.all((L(c.PDC())==0) == ((B0+I)==0)))
        self.assertFalse(np.all(L(c.PDC()) == L(c.PDC()).T))
        #     final sink should be 1 over all frequencies
        self.assertEqual(L(c.PDC())[0,0], Nfft)
        #     sources with equal outgoing connections should be equal
        self.assertEqual(L(c.PDC())[1,1], L(c.PDC())[2,2])
        #     equal connections in B should be equal,
        self.assertEqual(L(c.PDC())[0,1], L(c.PDC())[1,2])
        # ffPDC should have the same structure as B,
        self.assertTrue(np.all((L(c.ffPDC())==0) == ((B0+I)==0)))
        self.assertFalse(np.all(L(c.ffPDC()) == L(c.ffPDC()).T))
        #     sources with equal outgoing connections should be equal
        self.assertEqual(L(c.ffPDC())[1,1], L(c.ffPDC())[2,2])
        #     equal connections in B should be equal,
        self.assertEqual(L(c.ffPDC())[0,1], L(c.ffPDC())[1,2])
        # PDCF should equal PDC for identity noise covariance
        self.assertTrue(np.all(c.PDC() == c.PDCF()))
        # GPDC should equal PDC for identity noise covariance
        self.assertTrue(np.all(c.PDC() == c.GPDC()))
        # DTF should be upper triangular
        self.assertTrue(np.all(np.tril(K(c.DTF()), -1) == 0))
        self.assertFalse(np.all(K(c.DTF()) == K(c.DTF()).T))
        #     first source should be 1 over all frequencies
        self.assertEqual(L(c.DTF())[2,2], Nfft)
        # ffDTF should be upper triangular
        self.assertTrue(np.all(np.tril(K(c.ffDTF()), -1) == 0))
        self.assertFalse(np.all(K(c.ffDTF()) == K(c.ffDTF()).T))
        # dDTF should have the same structure as B,
        self.assertTrue(np.all((L(c.dDTF())==0) == ((B0+I)==0)))
        self.assertFalse(np.all(L(c.dDTF()) == L(c.dDTF()).T))
        # GDTF should equal DTF for identity noise covariance
        self.assertTrue(np.all(c.DTF() == c.GDTF()))
        
def main():
    unittest.main()

if __name__ == '__main__':
    main()
