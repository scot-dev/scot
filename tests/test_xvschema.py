# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013-2015 SCoT Development Team

import unittest

import numpy as np

import scot.xvschema


class Tests(unittest.TestCase):
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
