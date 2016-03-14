# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013-2015 SCoT Development Team

from __future__ import division

import unittest

import numpy as np

import scot
import scot.datatools
from scot import utils


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

    def test_cuthill(self):
        A = np.array([[0,0,1,1], [0,0,0,0], [1,0,1,0], [1,0,0,0]])
        p = scot.utils.cuthill_mckee(A)
        self.assertEqual(p, [1, 3, 0, 2])
