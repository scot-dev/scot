# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2014 SCoT Development Team

import unittest
import numpy as np

from numpy.testing.utils import assert_allclose

from scot.eegtopo.warp_layout import warp_locations
from scot.eegtopo.eegpos3d import positions as _eeglocs


eeglocs = [p.list for p in _eeglocs]


class TestWarpLocations(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_interface(self):
        self.assertRaises(TypeError, warp_locations)
        self.assertEqual(warp_locations(np.eye(3)).shape, (3, 3))  # returns array
        self.assertEqual(len(warp_locations(np.eye(3), return_ellipsoid=True)), 3)  # returns tuple

    def test_invariance(self):
        """unit-sphere locations should remain unchanged."""
        locs = [[1, 0, 0], [-1, 0, 0],
                [0, 1, 0], [0, -1, 0],
                [0, 0, 1], [0, 0, -1]]
        warp1 = warp_locations(locs)
        warp2, c, r = warp_locations(locs, return_ellipsoid=True)

        assert_allclose(warp1, locs, atol=1e-12)
        assert_allclose(warp2, locs, atol=1e-12)
        assert_allclose(c, 0, atol=1e-12)
        assert_allclose(r, 1, atol=1e-12)

    def test_eeglocations(self):
        np.random.seed(42)

        scale = np.random.rand(3) * 10 + 10
        displace = np.random.randn(3) * 100
        noise = np.random.randn(len(eeglocs), 3) * 5

        assert_allclose(warp_locations(eeglocs), eeglocs, atol=1e-10)
        assert_allclose(warp_locations(eeglocs * scale), eeglocs, atol=1e-10)
        assert_allclose(warp_locations(eeglocs * scale + displace), eeglocs, atol=1e-10)
        warp = warp_locations(eeglocs * scale + displace + noise)
        assert_allclose(np.sum(warp**2, axis=1), 1, atol=1e-12)  # all locations on unit shpere

