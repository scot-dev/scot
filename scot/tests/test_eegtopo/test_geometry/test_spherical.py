# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2014 SCoT Development Team

from __future__ import division

import unittest

import numpy as np

from scot.eegtopo.geo_euclidean import Vector
from scot.eegtopo.geo_spherical import Point, Line, Circle, Construct


class TestClasses(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testPoint(self):
        # Points must alway lie on unit sphere
        self.assertEqual(Point().vector.norm2(), 1)
        self.assertEqual(Point(1, 2, 3).vector.norm2(), 1)
        self.assertTrue(np.allclose(Point(10, -20, 30).vector.norm2(), 1))
        self.assertTrue(np.allclose(Point(100, 200, -300).vector.norm2(), 1))
        self.assertTrue(np.allclose(Point(-100000.0, 2, 300).vector.norm2(), 1))

        self.assertEqual(Point(1, 0, 0).distance(Point(0, 1, 0)), 0.5*np.pi)
        self.assertEqual(Point(1, 0, 0).distance(Point(0, 0, 1)), 0.5*np.pi)
        self.assertEqual(Point(0, 1, 0).distance(Point(0, 0, 1)), 0.5*np.pi)
        self.assertEqual(Point(0, 1, 0).distance(Point(1, 0, 0)), 0.5*np.pi)
        self.assertEqual(Point(0, 0, 1).distance(Point(1, 0, 0)), 0.5*np.pi)

        self.assertEqual(Point(1, 0, 0).distance(Point(-1, 0, 0)), np.pi)

    def testLine(self):
        self.assertTrue(Line(Point(1, 0, 0), Point(0, 1, 0)).get_point(0).vector.close(Vector(1, 0, 0)))
        self.assertTrue(Line(Point(1, 0, 0), Point(0, 1, 0)).get_point(1).vector.close(Vector(0, 1, 0)))
        self.assertTrue(Line(Point(1, 0, 0), Point(0, 1, 0)).get_point(2).vector.close(Vector(-1, 0, 0)))
        self.assertTrue(Line(Point(1, 0, 0), Point(0, 1, 0)).get_point(3).vector.close(Vector(0, -1, 0)))
        self.assertTrue(Line(Point(1, 0, 0), Point(0, 1, 0)).get_point(4).vector.close(Vector(1, 0, 0)))

        self.assertEqual(Line(Point(1, 0, 0), Point(0, 1, 0)).distance(Point(0, 0, 1)), 0.5*np.pi)

    def testCircle(self):
        self.assertEqual(Circle(Point(1, 0, 0), Point(0, 1, 0)).get_radius(), 0.5*np.pi) # circle radius measured on the surface
        self.assertEqual(Circle(Point(1, 0, 0), Point(0, 1, 0), Point(0, -1, 0)).get_radius(), 0.5*np.pi) # circle radius measured on the surface

        self.assertEqual(Circle(Point(1, 0, 0), Point(0, 1, 0)).angle(Point(0, 1, 0)), 0)
        self.assertEqual(Circle(Point(1, 0, 0), Point(0, 1, 0)).angle(Point(1, 0, 0)), 0.5*np.pi)
        self.assertEqual(Circle(Point(1, 0, 0), Point(0, 1, 0)).angle(Point(-1, 0, 0)), 0.5*np.pi)
        self.assertEqual(Circle(Point(1, 0, 0), Point(0, 1, 0)).angle(Point(0, -1, 0)), np.pi)

        self.assertEqual(Circle(Point(1, 0, 0), Point(0, 1, 0)).angle(Point(0, 0, -1)), 0.5*np.pi)
        self.assertEqual(Circle(Point(1, 0, 0), Point(0, 1, 0)).angle(Point(0, 0, -1)), 0.5*np.pi)

    def testConstruct(self):
        self.assertTrue(Construct.midpoint(Point(1, 0, 0), Point(-1, 1e-10, 0)).vector.close(Vector(0, 1, 0)))
        self.assertTrue(Construct.midpoint(Point(1, 0, 0), Point(0, 0, 1)).vector.close(Point(1, 0, 1).vector))

        a = Line(Point(1, 0, 0), Point(0, 1, 0))
        b = Line(Point(1, 0, 0), Point(0, 0, 1))
        ab = Construct.line_intersect_line(a, b)
        self.assertEqual(ab[0].vector, Vector(1, 0, 0))
        self.assertEqual(ab[1].vector, Vector(-1, 0, 0))

        a = Line(Point(1, 0, 0), Point(0, 1, 0))
        b = Line(Point(0, 0, 1), Point(0, 1, 0))
        c = Circle(Point(0, 1, 0), Point(1, 0, 0))
        ac = Construct.line_intersect_circle(a, c)
        bc = Construct.line_intersect_circle(b, c)
        self.assertEqual(ac, None)
        self.assertEqual(bc[0].vector, Vector(0, 0, 1))
        self.assertEqual(bc[1].vector, Vector(0, 0, -1))

        a = Circle(Point(1, 0, 0), Point(0, 1, 0))
        b = Circle(Point(0, 1, 0), Point(0, 0, 1))
        ab = Construct.circle_intersect_circle(a, b)
        self.assertEqual(ab[0].vector, Vector(0, 0, 1))
        self.assertEqual(ab[1].vector, Vector(0, 0, -1))

        a = Circle(Point(1, 0, 0), Point(0, 1, 0))
        b = Circle(Point(0, 1, 0), Point(0, 1, 1))
        ab = Construct.circle_intersect_circle(a, b)
        self.assertTrue(ab[0].vector.close(Point(0, 1, 1).vector))
        self.assertTrue(ab[1].vector.close(Point(0, 1, -1).vector))
