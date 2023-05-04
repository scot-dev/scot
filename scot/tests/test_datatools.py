# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013-2015 SCoT Development Team

from __future__ import division

import unittest
import numpy as np

from scot import datatools


class TestDataMangling(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_cut_epochs(self):
        triggers = [100, 200, 300, 400, 500, 600, 700, 800, 900]
        rawdata = np.random.randn(5, 1000)
        rawcopy = rawdata.copy()

        start, stop = -10, 50
        x = datatools.cut_segments(rawdata, triggers, start, stop)
        self.assertTrue(np.all(rawdata == rawcopy))
        self.assertEqual(x.shape, (len(triggers), rawdata.shape[0], stop - start))

        # test if it works with float indices
        start, stop = -10.0, 50.0
        x = datatools.cut_segments(rawdata, triggers, start, stop)
        self.assertEqual(x.shape, (len(triggers), x.shape[1], int(stop) - int(start)))

        self.assertRaises(ValueError, datatools.cut_segments,
                          rawdata, triggers, 0, 10.001)
        self.assertRaises(ValueError, datatools.cut_segments,
                          rawdata, triggers, -10.1, 50)

        for it in range(len(triggers)):
            a = rawdata[:, triggers[it] + int(start): triggers[it] + int(stop)]
            b = x[it, :, :]
            self.assertTrue(np.all(a == b))

    def test_cat_trials(self):
        x = np.random.randn(9, 5, 60)
        xc = x.copy()

        y = datatools.cat_trials(x)

        self.assertTrue(np.all(x == xc))
        self.assertEqual(y.shape, (x.shape[1], x.shape[0] * x.shape[2]))

        for it in range(x.shape[0]):
            a = y[:, it * x.shape[2]: (it + 1) * x.shape[2]]
            b = x[it, :, :]
            self.assertTrue(np.all(a == b))

    def test_dot_special(self):
        x = np.random.randn(9, 5, 60)
        a = np.eye(5) * 2.0

        xc = x.copy()
        ac = a.copy()

        y = datatools.dot_special(a, x)

        self.assertTrue(np.all(x == xc))
        self.assertTrue(np.all(a == ac))
        self.assertTrue(np.all(x * 2 == y))

        x = np.random.randn(150, 40, 6)
        a = np.ones((7, 40))
        y = datatools.dot_special(a, x)
        self.assertEqual(y.shape, (150, 7, 6))

    def test_acm_1d(self):
        """Test autocorrelation matrix for 1D input"""
        v = np.array([1, 2, 0, 0, 1, 2, 0, 0])
        acm = lambda l: datatools.acm(v, l)

        self.assertEqual(np.mean(v**2), acm(0))
        for l in range(1, 6):
            self.assertEqual(np.correlate(v[l:], v[:-l]) / (len(v) - l),
                             acm(l))


class TestRegressions(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_cat_trials_dimensions(self):
        """cat_trials did not always return a 2d array."""
        self.assertEqual(datatools.cat_trials(np.random.randn(2, 2, 100)).ndim, 2)
        self.assertEqual(datatools.cat_trials(np.random.randn(1, 2, 100)).ndim, 2)
        self.assertEqual(datatools.cat_trials(np.random.randn(2, 1, 100)).ndim, 2)
        self.assertEqual(datatools.cat_trials(np.random.randn(1, 1, 100)).ndim, 2)
