# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

import unittest

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import AxesImage
from matplotlib.figure import Figure

from scot.eegtopo.topoplot import Topoplot
from scot import plotting as sp
from scot.varbase import VARBase


class TestFunctionality(unittest.TestCase):
    def setUp(self):
        self.locs = [[0, 0, 1], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]]
        self.vals = [[1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1] ]

        self.topo = Topoplot()
        self.topo.set_locations(self.locs)
        self.maps = sp.prepare_topoplots(self.topo, self.vals)

    def tearDown(self):
        plt.close('all')

    def test_topoplots(self):
        locs, vals, topo, maps = self.locs, self.vals, self.topo, self.maps

        self.assertEquals(len(maps), len(vals))     # should get two topo maps

        self.assertTrue(np.allclose(maps[0], maps[0].T))    # first map: should be rotationally identical (blob in the middle)
        self.assertTrue(np.alltrue(maps[1] == 0))           # second map: should be all zeros
        self.assertTrue(np.alltrue(maps[2] == 1))           # third map: should be all ones

        #--------------------------------------------------------------------

        a1 = sp.plot_topo(plt.gca(), topo, maps[0])
        a2 = sp.plot_topo(plt.gca(), topo, maps[0], crange=[-1, 1], offset=(1, 1))

        self.assertIsInstance(a1, AxesImage)
        self.assertIsInstance(a2, AxesImage)

        #--------------------------------------------------------------------

        f1 = sp.plot_sources(topo, maps, maps)
        f2 = sp.plot_sources(topo, maps, maps, 90, f1)

        self.assertIs(f1, f2)
        self.assertIsInstance(f1, Figure)

        #--------------------------------------------------------------------

        f1 = sp.plot_connectivity_topos(topo=topo, topomaps=maps, layout='diagonal')
        f2 = sp.plot_connectivity_topos(topo=topo, topomaps=maps, layout='somethingelse')

        self.assertEqual(len(f1.axes), len(vals))
        self.assertEqual(len(f2.axes), len(vals)*2)

    def test_connectivity_spectrum(self):
        a = np.array([[[0, 0], [0, 1], [0, 2]],
                      [[1, 0], [1, 1], [1, 2]],
                      [[2, 0], [2, 1], [2, 2]]])
        f = sp.plot_connectivity_spectrum(a, diagonal=0)
        self.assertIsInstance(f, Figure)
        self.assertEqual(len(f.axes), 9)

        f = sp.plot_connectivity_spectrum(a, diagonal=1)
        self.assertEqual(len(f.axes), 3)

        f = sp.plot_connectivity_spectrum(a, diagonal=-1)
        self.assertEqual(len(f.axes), 6)

    def test_connectivity_significance(self):
        a = np.array([[[0, 0], [0, 1], [0, 2]],
                      [[1, 0], [1, 1], [1, 2]],
                      [[2, 0], [2, 1], [2, 2]]])
        f = sp.plot_connectivity_significance(a, diagonal=0)
        self.assertIsInstance(f, Figure)
        self.assertEqual(len(f.axes), 9)

        f = sp.plot_connectivity_significance(a, diagonal=1)
        self.assertEqual(len(f.axes), 3)

        f = sp.plot_connectivity_significance(a, diagonal=-1)
        self.assertEqual(len(f.axes), 6)

    def test_connectivity_timespectrum(self):
        a = np.array([[[[0, 0], [0, 1], [0, 2]],
                      [[1, 0], [1, 1], [1, 2]],
                      [[2, 0], [2, 1], [2, 2]]]]).repeat(4, 0).transpose([1,2,3,0])
        f = sp.plot_connectivity_timespectrum(a, diagonal=0)
        self.assertIsInstance(f, Figure)
        self.assertEqual(len(f.axes), 9)

        f = sp.plot_connectivity_timespectrum(a, diagonal=1)
        self.assertEqual(len(f.axes), 3)

        f = sp.plot_connectivity_timespectrum(a, diagonal=-1)
        self.assertEqual(len(f.axes), 6)

    def test_circular(self):
        w = [[1, 1, 1],
             [1, 1, 1],
             [1, 1, 1]]
        c = [[[1, 1, 1], [1, 1, 1], [1, 1, 1]],
             [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
             [[1, 1, 1], [1, 1, 1], [1, 1, 1]]]

        sp.plot_circular(1, [1, 1, 1], topo=self.topo, topomaps=self.maps)
        sp.plot_circular(w, [1, 1, 1], topo=self.topo, topomaps=self.maps)
        sp.plot_circular(1, c, topo=self.topo, topomaps=self.maps)
        sp.plot_circular(w, c, topo=self.topo, topomaps=self.maps)
        sp.plot_circular(w, c, mask=False, topo=self.topo, topomaps=self.maps)

    def test_whiteness(self):
        np.random.seed(91)

        var = VARBase(0)
        var.residuals = np.random.randn(10, 5, 100)

        pr = sp.plot_whiteness(var, 20, repeats=100)

        self.assertGreater(pr, 0.05)
