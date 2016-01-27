# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

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
        rawdata = np.random.randn(1000, 5)
        rawcopy = rawdata.copy()

        start, stop = -10, 50
        x = datatools.cut_segments(rawdata, triggers, start, stop)
        self.assertTrue(np.all(rawdata == rawcopy))
        self.assertEqual(x.shape, (stop - start, x.shape[1], len(triggers)))

        # test if it works with float indices
        start, stop = -10.0, 50.0
        x = datatools.cut_segments(rawdata, triggers, start, stop)
        self.assertEqual(x.shape, (stop - start, x.shape[1], len(triggers)))

        self.assertRaises(ValueError, datatools.cut_segments,
                          rawdata, triggers, 0, 10.001)
        self.assertRaises(ValueError, datatools.cut_segments,
                          rawdata, triggers, -10.1, 50)

        for it in range(len(triggers)):
            a = rawdata[triggers[it] + start: triggers[it] + stop, :]
            b = x[:, :, it]
            self.assertTrue(np.all(a == b))

    def test_cat_trials(self):
        x = np.random.randn(60, 5, 9)
        xc = x.copy()

        y = datatools.cat_trials(x)

        self.assertTrue(np.all(x == xc))
        self.assertEqual(y.shape, (x.shape[0] * x.shape[2], x.shape[1]))

        for it in range(x.shape[2]):
            a = y[it * x.shape[0]: (it + 1) * x.shape[0], :]
            b = x[:, :, it]
            self.assertTrue(np.all(a == b))

    def test_dot_special(self):
        x = np.random.randn(60, 5, 9)
        a = np.eye(5) * 2.0

        xc = x.copy()
        ac = a.copy()

        y = datatools.dot_special(x, a)

        self.assertTrue(np.all(x == xc))
        self.assertTrue(np.all(a == ac))
        self.assertTrue(np.all(x * 2 == y))


class TestRegressions(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_cat_trials_dimensions(self):
        """cat_trials did not always return a 2d array."""
        self.assertEqual(datatools.cat_trials(np.random.randn(100, 2, 2)).ndim, 2)
        self.assertEqual(datatools.cat_trials(np.random.randn(100, 1, 2)).ndim, 2)
        self.assertEqual(datatools.cat_trials(np.random.randn(100, 2, 1)).ndim, 2)
        self.assertEqual(datatools.cat_trials(np.random.randn(100, 1, 1)).ndim, 2)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
