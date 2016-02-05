# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2014-2015 SCoT Development Team

import unittest

import numpy as np

from scot.var import VAR
import scot.connectivity_statistics as cs


class TestFunctions(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    @staticmethod
    def generate_data():
        var = VAR(2)
        var.coef = np.array([[0.2, 0.1, 0, 0], [0.7, -0.4, 0.1, 0]])
        l = (100, 100)
        x = var.simulate(l)
        return x, var

    def test_surrogate(self):
        np.random.seed(31415)
        x, var0 = self.generate_data()

        result = cs.surrogate_connectivity('PDC', x, VAR(2), nfft=4,
                                           repeats=100)
        self.assertEqual(result.shape, (100, 2, 2, 4))

        structure = np.mean(np.mean(result, axis=3), axis=0)
        self.assertTrue(np.all(np.abs(structure-np.eye(2)) < 0.05))

    def test_jackknife(self):
        np.random.seed(31415)
        x, var0 = self.generate_data()

        result = cs.jackknife_connectivity('PDC', x, VAR(2), nfft=4,
                                           leaveout=1)
        self.assertEqual(result.shape, (100, 2, 2, 4))

        structure = np.mean(np.mean(result, axis=3), axis=0)
        # make sure result has roughly the correct structure
        self.assertTrue(np.all(np.abs(structure-[[1, 0], [0.5, 1]]) < 0.25))

    def test_bootstrap(self):
        np.random.seed(31415)
        x, var0 = self.generate_data()

        result = cs.bootstrap_connectivity('PDC', x, VAR(2), nfft=4,
                                           repeats=100)
        self.assertEqual(result.shape, (100, 2, 2, 4))

        structure = np.mean(np.mean(result, axis=3), axis=0)
        # make sure result has roughly the correct structure
        self.assertTrue(np.all(np.abs(structure - [[1, 0], [0.5, 1]]) < 0.25))

    def test_bootstrap_difference_and_fdr(self):
        # Generate reference data
        np.random.seed(31415)
        x, var0 = self.generate_data()
        a = cs.bootstrap_connectivity('PDC', x, VAR(2), nfft=4, repeats=100)

        # Similar to reference data ==> no significant differences expected
        np.random.seed(12345)
        x, var0 = self.generate_data()
        b = cs.bootstrap_connectivity('PDC', x, VAR(2), nfft=4, repeats=100)
        p = cs.test_bootstrap_difference(a, b)
        self.assertFalse(np.any(p < 0.01))  # TODO: np.all?
        self.assertFalse(np.any(cs.significance_fdr(p, 0.05)))  # TODO: np.all?

        # Trials rearranged ==> no significant differences expected
        np.random.seed(12345)
        x, var0 = self.generate_data()
        b = cs.bootstrap_connectivity('PDC', x[::-1, :, :], VAR(2), nfft=4,
                                      repeats=100)
        p = cs.test_bootstrap_difference(a, b)
        self.assertFalse(np.any(p < 0.01))
        self.assertFalse(np.any(cs.significance_fdr(p, 0.05)))

        # Channels rearranged ==> highly significant differences expected
        np.random.seed(12345)
        x, var0 = self.generate_data()
        b = cs.bootstrap_connectivity('PDC', x[1, ::-1, :], VAR(2), nfft=4,
                                      repeats=100)
        p = cs.test_bootstrap_difference(a, b)
        self.assertTrue(np.all(p < 0.0001))
        self.assertTrue(np.all(cs.significance_fdr(p, 0.01)))

        # Time reversed ==> highly significant differences expected
        np.random.seed(12345)
        x, var0 = self.generate_data()
        b = cs.bootstrap_connectivity('PDC', x[1, :, ::-1], VAR(2), nfft=4,
                                      repeats=100)
        p = cs.test_bootstrap_difference(a, b)
        self.assertTrue(np.all(p < 0.0001))
        self.assertTrue(np.all(cs.significance_fdr(p, 0.01)))
