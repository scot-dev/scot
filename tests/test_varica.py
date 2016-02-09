# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013-2015 SCoT Development Team

import unittest
from importlib import import_module

import numpy as np
from numpy.testing import assert_allclose

import scot
from scot import varica, datatools
from scot.var import VAR


class TestMVARICA(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testInterface(self):
        self.assertRaises(TypeError, varica.mvarica)
        # simply pass in different data shapes and see if the functions runs without error
        varica.mvarica(np.sin(np.arange(30)).reshape((10, 3)), VAR(1))    # 10 samples, 3 channels
        varica.mvarica(np.sin(np.arange(30)).reshape((5, 3, 2)), VAR(1))  # 5 samples, 3 channels, 2 trials

    def testModelIdentification(self):
        """ generate VAR signals, mix them, and see if MVARICA can reconstruct the signals
            do this for every backend """

        # original model coefficients
        b0 = np.zeros((3, 6))
        b0[1:3, 2:6] = [[0.4, -0.2, 0.3, 0.0],
                        [-0.7, 0.0, 0.9, 0.0]]
        m0 = b0.shape[0]
        l, t = 1000, 100

        # generate VAR sources with non-gaussian innovation process, otherwise ICA won't work
        noisefunc = lambda: np.random.normal(size=(1, m0)) ** 3 / 1e3

        var = VAR(2)
        var.coef = b0
        sources = var.simulate([l, t], noisefunc)

        # simulate volume conduction... 3 sources measured with 7 channels
        mix = [[0.5, 1.0, 0.5, 0.2, 0.0, 0.0, 0.0],
               [0.0, 0.2, 0.5, 1.0, 0.5, 0.2, 0.0],
               [0.0, 0.0, 0.0, 0.2, 0.5, 1.0, 0.5]]
        data = datatools.dot_special(np.transpose(mix), sources)

        for backend_name, backend_gen in scot.backend.items():

            # apply MVARICA
            #  - default setting of 0.99 variance should reduce to 3 channels with this data
            #  - automatically determine delta (enough data, so it should most likely be 0)
            result = varica.mvarica(data, var, optimize_var=True, backend=backend_gen())

            # ICA does not define the ordering and sign of components
            # so wee need to test all combinations to find if one of them fits the original coefficients
            permutations = np.array(
                [[0, 1, 2, 3, 4, 5], [0, 1, 4, 5, 2, 3], [2, 3, 4, 5, 0, 1], [2, 3, 0, 1, 4, 5], [4, 5, 0, 1, 2, 3],
                 [4, 5, 2, 3, 0, 1]])
            signperms = np.array(
                [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, -1, -1], [1, 1, -1, -1, 1, 1], [1, 1, -1, -1, -1, -1],
                 [-1, -1, 1, 1, 1, 1], [-1, -1, 1, 1, -1, -1], [-1, -1, -1, -1, 1, 1], [-1, -1, -1, -1, -1, -1]])

            best, d = np.inf, None

            for perm in permutations:
                b = result.b.coef[perm[::2] // 2, :]
                b = b[:, perm]
                for sgn in signperms:
                    c = b * np.repeat([sgn], 3, 0) * np.repeat([sgn[::2]], 6, 0).T
                    err = np.sum((c - b0) ** 2)
                    if err < best:
                        best = err
                        d = c

            assert_allclose(d, b0, rtol=1e-2, atol=1e-2)


class TestCSPVARICA(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testInterface(self):
        self.assertRaises(TypeError, varica.cspvarica)
        # simply pass in different data shapes and see if the functions runs without error
        self.assertRaises(AttributeError, varica.cspvarica, np.sin(np.arange(30)).reshape((10, 3)), VAR(1), [0])
        varica.cspvarica(np.sin(np.arange(30)).reshape((2, 3, 5)), VAR(1), ['A', 'B'])  # 5 samples, 3 channels, 2 trials


