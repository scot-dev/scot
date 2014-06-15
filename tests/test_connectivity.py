# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

import unittest

import numpy as np

from scot import connectivity


class TestFunctionality(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testSimple(self):
        # Three sources: a <- b <-c
        # simply test if connectivity measures are 0 where expected to be so
        b0 = np.array([[0, 0.9, 0], [0, 0, 0.9], [0, 0, 0]])
        identity = np.eye(3)
        nfft = 5
        c = connectivity.Connectivity(b=b0, c=identity, nfft=nfft)
        k = lambda x: np.sum(np.abs(x), 2)
        l = lambda x: np.sum(x, 2)
        # a should have the same structure as b
        self.assertTrue(np.all((k(c.A()) == 0) == ((b0 + identity) == 0)))
        self.assertFalse(np.all(k(c.A()) == k(c.A()).T))
        # H should be upper triangular
        self.assertTrue(np.all(np.tril(k(c.H()), -1) == 0))
        self.assertFalse(np.all(k(c.H()) == k(c.H()).T))
        # S should be a full matrix and symmetric
        self.assertTrue(np.all(k(c.S()) > 0))
        self.assertTrue(np.all(k(c.S()) == k(c.S()).T))
        # g should be nonzero for direct connections only and symmetric in magnitude
        self.assertEqual(k(c.G())[0, 2], 0)
        self.assertTrue(np.all(k(c.G()) == k(c.G()).T))
        # Phase should be zero along the diagonal
        self.assertTrue(np.all(k(c.PHI()).diagonal() == 0))
        # Phase should be zero along the diagonal and antisymmetric
        self.assertTrue(np.all(k(c.PHI()).diagonal() == 0))
        self.assertTrue(np.all(l(c.PHI()) == -l(c.PHI()).T))
        # Coherence should be 1 over all frequencies along the diagonal
        self.assertTrue(np.all(k(c.COH()).diagonal() == nfft))
        self.assertLessEqual(np.max(np.abs(c.COH())), 1)
        # pCOH should be nonzero for direct connections only and symmetric in magnitude
        self.assertEqual(k(c.pCOH())[0, 2], 0)
        self.assertTrue(np.all(k(c.pCOH()) == k(c.pCOH()).T))
        # PDC should have the same structure as b,
        self.assertTrue(np.all((l(c.PDC()) == 0) == ((b0 + identity) == 0)))
        self.assertFalse(np.all(l(c.PDC()) == l(c.PDC()).T))
        #     final sink should be 1 over all frequencies
        self.assertEqual(l(c.PDC())[0, 0], nfft)
        #     sources with equal outgoing connections should be equal
        self.assertEqual(l(c.PDC())[1, 1], l(c.PDC())[2, 2])
        #     equal connections in b should be equal,
        self.assertEqual(l(c.PDC())[0, 1], l(c.PDC())[1, 2])
        # ffPDC should have the same structure as b,
        self.assertTrue(np.all((l(c.ffPDC()) == 0) == ((b0 + identity) == 0)))
        self.assertFalse(np.all(l(c.ffPDC()) == l(c.ffPDC()).T))
        #     sources with equal outgoing connections should be equal
        self.assertEqual(l(c.ffPDC())[1, 1], l(c.ffPDC())[2, 2])
        #     equal connections in b should be equal,
        self.assertEqual(l(c.ffPDC())[0, 1], l(c.ffPDC())[1, 2])
        # PDCF should equal PDC for identity noise covariance
        self.assertTrue(np.all(c.PDC() == c.PDCF()))
        # GPDC should equal PDC for identity noise covariance
        self.assertTrue(np.all(c.PDC() == c.GPDC()))
        # DTF should be upper triangular
        self.assertTrue(np.all(np.tril(k(c.DTF()), -1) == 0))
        self.assertFalse(np.all(k(c.DTF()) == k(c.DTF()).T))
        #     first source should be 1 over all frequencies
        self.assertEqual(l(c.DTF())[2, 2], nfft)
        # ffDTF should be upper triangular
        self.assertTrue(np.all(np.tril(k(c.ffDTF()), -1) == 0))
        self.assertFalse(np.all(k(c.ffDTF()) == k(c.ffDTF()).T))
        # dDTF should have the same structure as b,
        self.assertTrue(np.all((l(c.dDTF()) == 0) == ((b0 + identity) == 0)))
        self.assertFalse(np.all(l(c.dDTF()) == l(c.dDTF()).T))
        # GDTF should equal DTF for identity noise covariance
        self.assertTrue(np.all(c.DTF() == c.GDTF()))


def main():
    unittest.main()


if __name__ == '__main__':
    main()
