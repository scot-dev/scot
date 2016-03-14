# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013-2015 SCoT Development Team

import unittest
from importlib import import_module

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from scot import datatools
import scot
from scot.var import VAR


class TestMVARICA(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testTrivia(self):
        api = scot.Workspace(VAR(1))
        str(api)

    def testExceptions(self):
        self.assertRaises(TypeError, scot.Workspace)
        api = scot.Workspace({'model_order':50})
        self.assertRaises(RuntimeError, api.remove_sources, [])
        self.assertRaises(RuntimeError, api.do_mvarica)
        self.assertRaises(RuntimeError, api.do_cspvarica)
        self.assertRaises(RuntimeError, api.do_ica)
        self.assertRaises(RuntimeError, api.fit_var)
        self.assertRaises(TypeError, api.get_connectivity)
        self.assertRaises(RuntimeError, api.get_connectivity, 'S')
        self.assertRaises(RuntimeError, api.get_tf_connectivity, 'PDC', 10, 1)
        api.set_data([[[1,1], [1,1]], [[1,1], [1,1]]])
        self.assertRaises(RuntimeError, api.do_cspvarica)
        
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

            api = scot.Workspace({'model_order': 2}, backend=backend_gen(),
                                 reducedim=0.99)

            api.set_data(data)

            # apply MVARICA
            #  - default setting of 0.99 variance should reduce to 3 channels with this data
            #  - automatically determine delta (enough data, so it should most likely be 0)
            api.do_mvarica()
            #result = varica.mvarica(data, 2, delta='auto', backend=bm.backend)

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
                b = api.var_.coef[perm[::2] // 2, :]
                b = b[:, perm]
                for sgn in signperms:
                    c = b * np.repeat([sgn], 3, 0) * np.repeat([sgn[::2]], 6, 0).T
                    err = np.sum((c - b0) ** 2)
                    if err < best:
                        best = err
                        d = c

            #self.assertTrue(np.all(abs(d - b0) < 0.05))
            assert_allclose(d, b0, rtol=1e-2, atol=2e-2)

    def testFunctionality(self):
        """ generate VAR signals, and apply the api to them
            do this for every backend """
        np.random.seed(3141592)

        # original model coefficients
        b01 = np.zeros((3, 6))
        b02 = np.zeros((3, 6))
        b01[1:3, 2:6] = [[0.4, -0.2, 0.3, 0.0],
                         [-0.7, 0.0, 0.9, 0.0]]
        b02[0:3, 2:6] = [[0.4, 0.0, 0.0, 0.0],
                         [0.4, 0.0, 0.4, 0.0],
                         [0.0, 0.0, 0.4, 0.0]]
        m0 = b01.shape[0]
        cl = np.array([0, 1, 0, 1, 0, 0, 1, 1, 1, 0])
        l = 200
        t = len(cl)

        # generate VAR sources with non-gaussian innovation process, otherwise ICA won't work
        noisefunc = lambda: np.random.normal(size=(1, m0)) ** 3 / 1e3

        var = VAR(2)
        var.coef = b01
        sources1 = var.simulate([l, sum(cl == 0)], noisefunc)
        var.coef = b02
        sources2 = var.simulate([l, sum(cl == 1)], noisefunc)

        var.fit(sources1)
        var.fit(sources2)

        sources = np.zeros((t, m0, l))

        sources[cl == 0, :, :] = sources1
        sources[cl == 1, :, :] = sources2

        # simulate volume conduction... 3 sources smeared over 7 channels
        mix = [[0.5, 1.0, 0.5, 0.2, 0.0, 0.0, 0.0],
               [0.0, 0.2, 0.5, 1.0, 0.5, 0.2, 0.0],
               [0.0, 0.0, 0.0, 0.2, 0.5, 1.0, 0.5]]
        data = datatools.dot_special(np.transpose(mix), sources)
        data += np.random.randn(*data.shape) * 0.001  # add small noise

        for backend_name, backend_gen in scot.backend.items():
            np.random.seed(3141592)  # reset random seed so we're independent of module order

            api = scot.Workspace({'model_order': 2}, reducedim=3, backend=backend_gen())

            api.set_data(data)

            api.do_ica()

            self.assertEqual(api.mixing_.shape, (3, 7))
            self.assertEqual(api.unmixing_.shape, (7, 3))

            api.do_mvarica()

            self.assertEqual(api.get_connectivity('S').shape, (3, 3, 512))

            self.assertFalse(np.any(np.isnan(api.activations_)))
            self.assertFalse(np.any(np.isinf(api.activations_)))

            api.set_data(data)

            api.fit_var()

            self.assertEqual(api.get_connectivity('S').shape, (3, 3, 512))
            self.assertEqual(api.get_tf_connectivity('S', 100, 50).shape, (3, 3, 512, (l-100)//50))
            
            tfc1 = api.get_tf_connectivity('PDC', 100, 5, baseline=None)        # no baseline
            tfc2 = api.get_tf_connectivity('PDC', 100, 5, baseline=[110, -10])  # invalid baseline
            tfc3 = api.get_tf_connectivity('PDC', 100, 5, baseline=[0, 0])      # one-window baseline
            tfc4 = tfc1 - tfc1[:, :, :, [0]]
            tfc5 = api.get_tf_connectivity('PDC', 100, 5, baseline=[-np.inf, np.inf])  # full trial baseline
            tfc6 = tfc1 - np.mean(tfc1, axis=3, keepdims=True)
            self.assertTrue(np.allclose(tfc1, tfc2))
            self.assertTrue(np.allclose(tfc3, tfc4))
            self.assertTrue(np.allclose(tfc5, tfc6, rtol=1e-05, atol=1e-06))

            api.set_data(data, cl)
            
            self.assertFalse(np.any(np.isnan(api.data_)))
            self.assertFalse(np.any(np.isinf(api.data_)))
            
            api.do_cspvarica()
            
            self.assertEqual(api.get_connectivity('S').shape, (3,3,512))

            self.assertFalse(np.any(np.isnan(api.activations_)))
            self.assertFalse(np.any(np.isinf(api.activations_)))
            
            for c in np.unique(cl):
                api.set_used_labels([c])

                api.fit_var()
                fc = api.get_connectivity('S')
                self.assertEqual(fc.shape, (3, 3, 512))

                tfc = api.get_tf_connectivity('S', 100, 50)
                self.assertEqual(tfc.shape, (3, 3, 512, (l-100)//50))

            api.set_data(data)
            api.remove_sources([0, 2])
            api.fit_var()
            self.assertEqual(api.get_connectivity('S').shape, (1, 1, 512))
            self.assertEqual(api.get_tf_connectivity('S', 100, 50).shape, (1, 1, 512, (l-100)//50))

            try:
                api.optimize_var()
            except NotImplementedError:
                pass
            api.fit_var()
            self.assertEqual(api.get_connectivity('S').shape, (1, 1, 512))
            self.assertEqual(api.get_tf_connectivity('S', 100, 50).shape, (1, 1, 512, (l-100)//50))

    def test_premixing(self):
        api = scot.Workspace(VAR(1))
        api.set_premixing([[0, 1], [1, 0]])

    def test_plotting(self):
        np.random.seed(3141592)

        api = scot.Workspace(VAR(1), locations=[[0, 0, 1], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]])

        api.set_data(np.random.randn(10, 5, 10), [1, 0] * 5)
        api.do_mvarica()

        api.plot_source_topos()

        for diag in ['S', 'fill', 'topo']:
            for outside in [True, False]:
                api.plot_diagonal = diag
                api.plot_outside_topo = outside

                fig = api.plot_connectivity_topos()
                api.get_connectivity('PHI', plot=fig)
                api.get_surrogate_connectivity('PHI', plot=fig, repeats=5)
                api.get_bootstrap_connectivity('PHI', plot=fig, repeats=5)
                api.get_tf_connectivity('PHI', winlen=2, winstep=1, plot=fig)
                api.compare_conditions([0], [1], 'PHI', plot=fig, repeats=5)

    def test_random_state(self):
        np.random.seed(10)
        api = scot.Workspace(VAR(1),locations=[[0, 0, 1], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]], reducedim=None)
        api.set_data(np.random.randn(10, 5, 10), [1, 0] * 5)

        # test MVARICA
        api.do_mvarica(random_state=1)
        mixing1 = api.mixing_
        api.do_mvarica(random_state=1)
        mixing2 = api.mixing_
        assert_array_equal(mixing1, mixing2)

        # test CSPVARICA
        api.do_cspvarica(random_state=1)
        mixing1 = api.mixing_
        api.do_cspvarica(random_state=1)
        mixing2 = api.mixing_
        assert_array_equal(mixing1, mixing2)

    def test_source_selection(self):
        var = VAR(2)
        var.coef = np.random.randn(16, 4)
        x = var.simulate([500, 50],
                         lambda: np.random.randn(16).dot(np.eye(16, 16)))
        api = scot.Workspace({'model_order': 2})
        api.set_data(x)
        self.assertRaises(RuntimeError, api.keep_sources, [0, 5, 11, 12])
        self.assertRaises(RuntimeError, api.remove_sources, [1, 2, 8, 14])

        # keep sources
        api.do_mvarica()
        api.keep_sources([0, 5, 11, 12])
        self.assertEqual(api.mixing_.shape, (4, 16))
        self.assertEqual(api.unmixing_.shape, (16, 4))

        # remove sources
        api.do_mvarica()
        api.remove_sources([1, 2, 8, 14])
        self.assertEqual(api.mixing_.shape, (12, 16))
        self.assertEqual(api.unmixing_.shape, (16, 12))

    def testBackendRegression(self):
        """Regression test for Github issue #103."""
        ws = scot.Workspace({'model_order': 3}, backend=None)
        self.assertIsNotNone(ws.backend_)
