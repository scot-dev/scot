# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

import unittest
from importlib import import_module
import numpy as np

import scot.backend
from scot import varica, datatools

from scot.builtin.var import VAR


class TestMVARICA(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

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
        noisefunc = lambda: np.random.normal(size=(1, m0)) ** 3

        var = VAR(2)
        var.coef = b0
        sources = var.simulate([l, t], noisefunc)

        # simulate volume conduction... 3 sources measured with 7 channels
        mix = [[0.5, 1.0, 0.5, 0.2, 0.0, 0.0, 0.0],
               [0.0, 0.2, 0.5, 1.0, 0.5, 0.2, 0.0],
               [0.0, 0.0, 0.0, 0.2, 0.5, 1.0, 0.5]]
        data = datatools.dot_special(sources, mix)

        backend_modules = [import_module('scot.backend.' + b) for b in scot.backend.__all__]

        for bm in backend_modules:

            # apply MVARICA
            #  - default setting of 0.99 variance should reduce to 3 channels with this data
            #  - automatically determine delta (enough data, so it should most likely be 0)
            result = varica.mvarica(data, var, optimize_var=True, backend=bm.backend)

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

            self.assertTrue(np.all(abs(d - b0) < 0.05))


def main():
    unittest.main()


if __name__ == '__main__':
    main()
