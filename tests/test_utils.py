# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

import unittest

from importlib import import_module
import numpy as np

import scot.backend

backend_modules = [import_module('scot.backend.' + b) for b in scot.backend.__all__]


def generate_backend_test(module):
    class BackendCase(unittest.TestCase):
        def setUp(self):
            pass

        def tearDown(self):
            pass

        def test_cartesian(self):
            cartesian = module.backend['utils'].cartesian
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
    globals()[testname] = generate_backend_test(bm)