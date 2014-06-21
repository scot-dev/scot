# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2014 SCoT Development Team

import unittest


def f(x):
    return x**2 - 1


def g(x, y, z):
    return x**y - z


def h(x):
    return x, x**2


class TestFunctions(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_parallel_loop(self):
        from scot.parallel import parallel_loop

        verbose = 0

        # reference list comprehension
        ref = [f(i) for i in range(10)]
        reg = [g(i, j, 5) for i, j in enumerate(range(10, 20))]
        reh = [h(i) for i in range(10)]

        # test non-parallel execution
        par, func = parallel_loop(f, n_jobs=None, verbose=verbose)
        self.assertEqual(ref, par(func(i) for i in range(10)))
        # multiple arguments
        par, func = parallel_loop(g, n_jobs=None, verbose=verbose)
        self.assertEqual(reg, par(func(i, j, 5) for i, j in enumerate(range(10, 20))))
        # multiple return values
        par, func = parallel_loop(h, n_jobs=None, verbose=verbose)
        self.assertEqual(reh, par(func(i) for i in range(10)))

        # test non-parallel execution with joblib
        par, func = parallel_loop(f, n_jobs=1, verbose=verbose)
        b = par(func(i) for i in range(10))
        self.assertEqual(ref, par(func(i) for i in range(10)))
        # multiple arguments
        par, func = parallel_loop(g, n_jobs=1, verbose=verbose)
        self.assertEqual(reg, par(func(i, j, 5) for i, j in enumerate(range(10, 20))))
        # multiple return values
        par, func = parallel_loop(h, n_jobs=1, verbose=verbose)
        self.assertEqual(reh, par(func(i) for i in range(10)))

        # test parallel execution with joblib
        par, func = parallel_loop(f, n_jobs=2, verbose=verbose)
        b = par(func(i) for i in range(10))
        self.assertEqual(ref, par(func(i) for i in range(10)))
        # multiple arguments
        par, func = parallel_loop(g, n_jobs=2, verbose=verbose)
        self.assertEqual(reg, par(func(i, j, 5) for i, j in enumerate(range(10, 20))))
        # multiple return values
        par, func = parallel_loop(h, n_jobs=2, verbose=verbose)
        self.assertEqual(reh, par(func(i) for i in range(10)))

        # test parallel execution with joblib
        par, func = parallel_loop(f, n_jobs=-1, verbose=verbose)
        b = par(func(i) for i in range(10))
        self.assertEqual(ref, par(func(i) for i in range(10)))
        # multiple arguments
        par, func = parallel_loop(g, n_jobs=-1, verbose=verbose)
        self.assertEqual(reg, par(func(i, j, 5) for i, j in enumerate(range(10, 20))))
        # multiple return values
        par, func = parallel_loop(h, n_jobs=-1, verbose=verbose)
        self.assertEqual(reh, par(func(i) for i in range(10)))

        # test parallel execution with joblib
        par, func = parallel_loop(f, n_jobs=10, verbose=verbose)
        b = par(func(i) for i in range(10))
        self.assertEqual(ref, par(func(i) for i in range(10)))
        # multiple arguments
        par, func = parallel_loop(g, n_jobs=10, verbose=verbose)
        self.assertEqual(reg, par(func(i, j, 5) for i, j in enumerate(range(10, 20))))
        # multiple return values
        par, func = parallel_loop(h, n_jobs=10, verbose=verbose)
        self.assertEqual(reh, par(func(i) for i in range(10)))


if __name__ == '__main__':
    unittest.main()