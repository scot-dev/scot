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

    def testFit(self):
        """ Test submodel fitting on instationary data
        """
        np.random.seed(42)

        # original model coefficients
        b01 = np.array([[0.0, 0], [0, 0]])
        b02 = np.array([[0.5, 0.3], [0.3, 0.5]])
        b03 = np.array([[0.1, 0.1], [0.1, 0.1]])
        t, m, l = 10, 2, 100

        noisefunc = lambda: np.random.normal(size=(1, m)) ** 3 / 1e3

        var = VAR(1)
        var.coef = b01
        sources1 = var.simulate([l, t], noisefunc)
        var.coef = b02
        sources2 = var.simulate([l, t], noisefunc)
        var.coef = b03
        sources3 = var.simulate([l, t * 2], noisefunc)

        sources = np.vstack([sources1, sources2, sources3])
        cl = [1] * t + [2] * t + [1, 2] * t

        var = VAR(1)
        r_trial = varica.mvarica(sources, var, cl, reducedim='no_pca', varfit='trial')
        r_class = varica.mvarica(sources, var, cl, reducedim='no_pca', varfit='class')
        r_ensemble = varica.mvarica(sources, var, cl, reducedim='no_pca', varfit='ensemble')

        vars = [np.var(r.var_residuals) for r in [r_trial, r_class, r_ensemble]]

        # class one consists of trials generated with b01 and b03
        # class two consists of trials generated with b02 and b03
        #
        # ensemble fitting cannot resolve any model -> highest residual variance
        # class fitting cannot only resolve (b01+b03) vs (b02+b03) -> medium residual variance
        # trial fitting can resolve all three models -> lowest residual variance

        self.assertLess(vars[0], vars[1])
        self.assertLess(vars[1], vars[2])

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

    def testFit(self):
        """ Test submodel fitting on instationary data
        """
        np.random.seed(42)

        # original model coefficients
        b01 = np.array([[0.0, 0], [0, 0]])
        b02 = np.array([[0.5, 0.3], [0.3, 0.5]])
        b03 = np.array([[0.1, 0.1], [0.1, 0.1]])
        t, m, l = 10, 2, 100

        noisefunc = lambda: np.random.normal(size=(1, m)) ** 3 / 1e3

        var = VAR(1)
        var.coef = b01
        sources1 = var.simulate([l, t], noisefunc)
        var.coef = b02
        sources2 = var.simulate([l, t], noisefunc)
        var.coef = b03
        sources3 = var.simulate([l, t * 2], noisefunc)

        sources = np.vstack([sources1, sources2, sources3])
        cl = [1] * t + [2] * t + [1, 2] * t

        var = VAR(1)
        r_trial = varica.cspvarica(sources, var, cl, reducedim=None, varfit='trial')
        r_class = varica.cspvarica(sources, var, cl, reducedim=None, varfit='class')
        r_ensemble = varica.cspvarica(sources, var, cl, reducedim=None, varfit='ensemble')

        vars = [np.var(r.var_residuals) for r in [r_trial, r_class, r_ensemble]]

        # class one consists of trials generated with b01 and b03
        # class two consists of trials generated with b02 and b03
        #
        # ensemble fitting cannot resolve any model -> highest residual variance
        # class fitting cannot only resolve (b01+b03) vs (b02+b03) -> medium residual variance
        # trial fitting can resolve all three models -> lowest residual variance
        print(vars)

        self.assertLess(vars[0], vars[1])
        self.assertLess(vars[1], vars[2])


