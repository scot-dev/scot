# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013-2015 SCoT Development Team

import unittest

import numpy as np
from numpy.testing import assert_array_equal

import scot.xvschema


class TestBuiltin(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_singletrial(self):
        n_trials = 10
        xv = scot.xvschema.singletrial(n_trials)
        for n, (train, test) in enumerate(xv):
            self.assertEqual(len(train), 1)
            self.assertEqual(len(test), n_trials - 1)

            for t in train:
                self.assertTrue(t not in test)

            self.assertEqual(train[0], n)

    def test_multitrial(self):
        n_trials = 10
        xv = scot.xvschema.multitrial(n_trials)
        for n, (train, test) in enumerate(xv):
            self.assertEqual(len(test), 1)
            self.assertEqual(len(train), n_trials - 1)

            for t in train:
                self.assertTrue(t not in test)

            self.assertEqual(test[0], n)

    def test_splitset(self):
        n_trials = 10
        xv = scot.xvschema.splitset(n_trials)
        for n, (train, test) in enumerate(xv):
            self.assertEqual(len(test), n_trials // 2)
            self.assertEqual(len(train), n_trials // 2)

            for t in train:
                self.assertTrue(t not in test)

    def test_nfold(self):
        n_trials = 50
        n_blocks = 5
        xv = scot.xvschema.make_nfold(n_blocks)(n_trials)
        for n, (train, test) in enumerate(xv):
            self.assertEqual(len(test), n_trials // n_blocks)
            self.assertEqual(len(train), n_trials - n_trials // n_blocks)

            for t in train:
                self.assertTrue(t not in test)
        self.assertEqual(n + 1, n_blocks)


class TestSklearn(unittest.TestCase):
    def setUp(self):
        try:
            import sklearn
        except ImportError:
            self.skipTest("could not import scikit-learn")

    def tearDown(self):
        pass

    def test_leave1out(self):
        from sklearn.cross_validation import LeaveOneOut
        n_trials = 10
        xv1 = scot.xvschema.multitrial(n_trials)
        xv2 = LeaveOneOut(n_trials)
        self._comparexv(xv1, xv2)

    def test_kfold(self):
        from sklearn.cross_validation import KFold
        n_trials = 15
        n_blocks = 5
        xv1 = scot.xvschema.make_nfold(n_blocks)(n_trials)
        xv2 = KFold(n_trials, n_folds=n_blocks, shuffle=False)
        self._comparexv(xv1, xv2)

    def test_application(self):
        from scot.var import VAR
        from sklearn.cross_validation import LeaveOneOut, KFold
        np.random.seed(42)
        x = np.random.randn(10, 3, 15)

        var = VAR(3, xvschema=lambda n, _: LeaveOneOut(n)).optimize_delta_bisection(x)
        self.assertGreater(var.delta, 0)
        var = VAR(3, xvschema=lambda n, _: KFold(n, 5)).optimize_delta_bisection(x)
        self.assertGreater(var.delta, 0)

    def _comparexv(self, xv1, xv2):
        for (a, b), (c, d) in zip(xv1, xv2):
            assert_array_equal(a, c)
            assert_array_equal(b, d)
