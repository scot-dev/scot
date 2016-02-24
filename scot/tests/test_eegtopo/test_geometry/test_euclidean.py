# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2014 SCoT Development Team

from __future__ import division

import unittest
from math import sqrt

import numpy as np

from scot.eegtopo.geo_euclidean import Vector


class TestVector(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_access(self):
        # Test __iter__ and __init__

        self.assertEqual(list(Vector()), [0, 0, 0])
        self.assertEqual(list(Vector(1, 2, 3)), [1, 2, 3])
        self.assertEqual(list(Vector(x=-4, y=-5, z=-6)), [-4, -5, -6])

        # Test alternative initialization
        self.assertEqual(list(Vector.fromiterable([7, 8, 9])), [7, 8, 9])
        self.assertEqual(list(Vector.fromvector(Vector(1, 2, 3))), [1, 2, 3])

        # Test __getitem__
        self.assertEqual(Vector(x=1)[0], 1)
        self.assertEqual(Vector(y=1)[1], 1)
        self.assertEqual(Vector(z=1)[2], 1)

        # Test __getattr__
        self.assertEqual(Vector(x=1).x, 1)
        self.assertEqual(Vector(y=1).y, 1)
        self.assertEqual(Vector(z=1).z, 1)

        # Test item assignment
        v = Vector()
        v[0], v[1], v[2] = 3, 4, 5
        self.assertEqual(list(v), [3, 4, 5])

        v.x, v.y, v.z = 6, 7, 8
        self.assertEqual(list(v), [6, 7, 8])

        # Test __repr__
        self.assertEqual(eval(repr(Vector(1, 2, 3))), Vector(1, 2, 3))

        # Basic Math
        self.assertEqual(Vector(1, 2, 3) + Vector(4, 5, 6), Vector(5, 7, 9))
        self.assertEqual(Vector(4, 5, 6) - Vector(1, 2, 3), Vector(3, 3, 3))
        self.assertEqual(Vector(1, 2, 3) * Vector(5, 4, 3), Vector(5, 8, 9))
        self.assertEqual(Vector(9, 8, 7) / Vector(3, 2, 1), Vector(3, 4, 7))
        self.assertEqual(Vector(1, 2, 3) + 1, Vector(2, 3, 4))
        self.assertEqual(Vector(4, 5, 6) - 1, Vector(3, 4, 5))
        self.assertEqual(Vector(1, 2, 3) * 2, Vector(2, 4, 6))
        self.assertEqual(Vector(4, 5, 6) / 2, Vector(2, 2.5, 3))

        # Inplace Math
        v = Vector(1, 1, 1)
        v += Vector(1, 2, 3)
        self.assertEqual(v, Vector(2, 3, 4))
        v -= Vector(-1, 1, 1)
        self.assertEqual(v, Vector(3, 2, 3))
        v *= Vector(1, 2, 3)
        self.assertEqual(v, Vector(3, 4, 9))
        v /= Vector(3, 2, 3)
        self.assertEqual(v, Vector(1, 2, 3))
        v -= 1
        self.assertEqual(v, Vector(0, 1, 2))
        v += 2
        self.assertEqual(v, Vector(2, 3, 4))
        v *= 2
        self.assertEqual(v, Vector(4, 6, 8))
        v /= 2
        self.assertEqual(v, Vector(2, 3, 4))

        # Vector Math
        self.assertEqual(Vector(1, 2, 3).dot(Vector(2, 2, 2)), 12)
        self.assertEqual(Vector(2, 0, 0).cross(Vector(0, 3, 0)), Vector(0, 0, 6))
        self.assertEqual(Vector(1, 2, 3).norm2(), 14)
        self.assertEqual(Vector(1, 2, 3).norm(), sqrt(14))
        self.assertTrue(np.allclose(Vector(8, 3, 9).normalize().norm2(), 1))
        self.assertTrue(np.allclose(Vector(-3, 1, 0).normalized().norm2(), 1))

        v = Vector(1, 0, 0)
        self.assertTrue(v.rotated(0.0*np.pi, Vector(0, 0, 1)).close(Vector(1, 0, 0)))
        self.assertTrue(v.rotated(0.5*np.pi, Vector(0, 0, 1)).close(Vector(0, 1, 0)))
        self.assertTrue(v.rotated(1.0*np.pi, Vector(0, 0, 1)).close(Vector(-1, 0, 0)))
        self.assertTrue(v.rotated(1.5*np.pi, Vector(0, 0, 1)).close(Vector(0, -1, 0)))
        self.assertTrue(v.rotated(2.0*np.pi, Vector(0, 0, 1)).close(Vector(1, 0, 0)))

        self.assertTrue(v.rotate(0.5*np.pi, Vector(0, 0, 1)).close(Vector(0, 1, 0)))
        self.assertTrue(v.rotate(0.5*np.pi, Vector(0, 0, 1)).close(Vector(-1, 0, 0)))
        self.assertTrue(v.rotate(0.5*np.pi, Vector(0, 0, 1)).close(Vector(0, -1, 0)))
        self.assertTrue(v.rotate(0.5*np.pi, Vector(0, 0, 1)).close(Vector(1, 0, 0)))
