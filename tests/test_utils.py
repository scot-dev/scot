# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

import unittest

from importlib import import_module
import numpy as np

import scot
from scot import utils

backend_modules = [import_module('scot.' + b) for b in scot.backends]


class TestUtils(unittest.TestCase):
        def setUp(self):
            pass

        def tearDown(self):
            pass

        def test_memoize(self):
            class Obj(object):
                @scot.utils.memoize
                def add_to(self, arg):
                    return self + arg
            self.assertRaises(TypeError, Obj.add_to, 1)
            self.assertEqual(3, Obj.add_to(1, 2))
            self.assertEqual(3, Obj.add_to(1, 2))

            class Obj(object):
                @scot.utils.memoize
                def squareone(self, a):
                    return a * a + 1
            obj = Obj()
            self.assertEqual(2, obj.squareone(1))
            self.assertEqual(2, obj.squareone(1))
            self.assertEqual(5, obj.squareone(2))
            self.assertEqual(10, obj.squareone(3))
            self.assertEqual(5, obj.squareone(2))
            self.assertEqual(10, obj.squareone(3))

        def test_acm(self):
            v = np.array([1, 2, 0, 0]*2)
            acm = lambda l: scot.utils.acm(v, l)
            self.assertEqual(np.mean(v**2), acm(0))
            self.assertEqual(0.5, acm(1))
            self.assertEqual(0, acm(2))
            self.assertEqual(0.25, acm(3))
            self.assertEqual(acm(0)*0.5, acm(4))
            self.assertEqual(acm(1)*0.5, acm(5))
            self.assertEqual(0, acm(6))

        def test_cuthill(self):
            A = np.array([[0,0,1,1], [0,0,0,0], [1,0,1,0], [1,0,0,0]])
            p = scot.utils.cuthill_mckee(A)
            self.assertEqual(p, [1, 3, 0, 2])


def generate_backend(module):
    class BackendCase(unittest.TestCase):
        def setUp(self):
            pass

        def tearDown(self):
            pass

        def test_cartesian(self):
            cartesian = utils.cartesian
            ret = cartesian(([1, 2, 3], [4, 5], [6, 7]))
            self.assertTrue(np.all(ret == np.array(
                [[1, 4, 6],
                 [1, 4, 7],
                 [1, 5, 6],
                 [1, 5, 7],
                 [2, 4, 6],
                 [2, 4, 7],
                 [2, 5, 6],
                 [2, 5, 7],
                 [3, 4, 6],
                 [3, 4, 7],
                 [3, 5, 6],
                 [3, 5, 7]])))
    return BackendCase


for bm in backend_modules:
    testname = 'TestBackend_' + bm.__name__.split('.')[-1]
    globals()[testname] = generate_backend(bm)


if __name__ == '__main__':
    unittest.main()