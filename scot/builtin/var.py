# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

""" vector autoregressive (VAR) model implementation """

import numpy as np
import scipy as sp
from ..var import VARBase
from ..datatools import cat_trials
from .. import xvschema as xv


class VAR(VARBase):
    def __init__(self, model_order, delta=0, xvschema=xv.multitrial):
        """ Create a new VAR model instance.

            Parameters     Default  Shape   Description
            --------------------------------------------------------------------------
            model_order    :      :       : Autoregressive model order
            delta          : 0    :       : Ridge parameter (regularization). If set to
                                            0, ordinary least squares fitting is performed.
            xvschema       :      : func  : Function to generate training and testing
                                            set for cross-validation based parameter
                                            optimization. See xvschema module.
        """
        VARBase.__init__(self, model_order)
        self.delta = delta
        self.xvschema = xvschema

    def fit(self, data):
        data = sp.atleast_3d(data)

        if self.delta == 0 or self.delta is None:
            # ordinary least squares
            (x, y) = self._construct_eqns(data)
        else:
            # regularized least squares (ridge regression)
            (x, y) = self._construct_eqns_rls(data)

        (b, res, rank, s) = sp.linalg.lstsq(x, y)

        self.coef = b.transpose()

        self.residuals = data - self.predict(data)
        self.rescov = sp.cov(cat_trials(self.residuals), rowvar=False)

        return self

    def optimize(self, data):
        self.optimize_delta_bisection(data, 100)
        return self


    def optimize_delta_bisection(self, data, skipstep=1):
        """ Use the bisection method to find optimal regularization parameter delta.

            Behavior of this function depends on the xvschema attribute.

            Parameters     Default  Shape   Description
            --------------------------------------------------------------------------
            data           :      : n,m,t : 3d data matrix (n samples, m signals, t trials)
                                  : n,m   : 2d data matrix (n samples, m signals)
            skipstep       : 1    : 1     : Higher values speed up the calculation but
                                            cause higher variance in cost function which
                                            will result in less accurate results.
        """
        data = sp.atleast_3d(data)
        (l, m, t) = data.shape
        assert (t > 1)

        maxsteps = 10
        maxdelta = 1e50

        a = -10
        b = 10

        transform = lambda x: sp.sqrt(sp.exp(x))

        msge = self._get_msge_with_gradient_func(data.shape)

        (ja, ka) = msge(data, transform(a), self.xvschema, skipstep)
        (jb, kb) = msge(data, transform(b), self.xvschema, skipstep)

        # before starting the real bisection, make sure the interval actually contains 0
        while sp.sign(ka) == sp.sign(kb):
            print('Bisection initial interval (%f,%f) does not contain zero. New interval: (%f,%f)' % (a, b, a * 2, b * 2))
            a *= 2
            b *= 2
            (jb, kb) = msge(data, transform(b), self.xvschema, skipstep)

            if transform(b) >= maxdelta:
                print('Bisection: could not find initial interval.')
                print(' ********* Delta set to zero! ************ ')
                return 0

        nsteps = 0

        while nsteps < maxsteps:

            # point where the line between a and b crosses zero
            # this is not very stable!
            #c = a + (b-a) * np.abs(ka) / np.abs(kb-ka)
            c = (a + b) / 2
            (j, k) = msge(data, transform(c), self.xvschema, skipstep)
            if sp.sign(k) == sp.sign(ka):
                a, ka = c, k
            else:
                b, kb = c, k

            nsteps += 1
            tmp = transform([a, b, a + (b - a) * np.abs(ka) / np.abs(kb - ka)])
            print('%d Bisection Interval: %f - %f, (projected: %f)' % (nsteps, tmp[0], tmp[1], tmp[2]))

        self.delta = transform(a + (b - a) * np.abs(ka) / np.abs(kb - ka))
        print('Final point: %f' % self.delta)
        return self

    def _construct_eqns_rls(self, data):
        """Construct VAR equation system with RLS constraint"""
        (l, m, t) = sp.shape(data)
        n = (l - self.p) * t     # number of linear relations
        # Construct matrix x (predictor variables)
        x = sp.zeros((n + m * self.p, m * self.p))
        for i in range(m):
            for k in range(1, self.p + 1):
                x[:n, i * self.p + k - 1] = sp.reshape(data[self.p - k:-k, i, :], n)
        sp.fill_diagonal(x[n:, :], self.delta)

        # Construct vectors yi (response variables for each channel i)
        y = sp.zeros((n + m * self.p, m))
        for i in range(m):
            y[:n, i] = sp.reshape(data[self.p:, i, :], n)

        return x, y

    def _msge_with_gradient_underdetermined(self, data, delta, xvschema, skipstep):
        (l, m, t) = data.shape
        d = None
        j, k = 0, 0
        nt = sp.ceil(t / skipstep)
        for s in range(0, t, skipstep):
            trainset, testset = xvschema(s, t)

            (a, b) = self._construct_eqns(sp.atleast_3d(data[:, :, trainset]))
            (c, d) = self._construct_eqns(sp.atleast_3d(data[:, :, testset]))

            e = sp.linalg.inv(sp.eye(a.shape[0]) * delta ** 2 + a.dot(a.transpose()))

            cc = c.transpose().dot(c)

            be = b.transpose().dot(e)
            bee = be.dot(e)
            bea = be.dot(a)
            beea = bee.dot(a)
            beacc = bea.dot(cc)
            dc = d.transpose().dot(c)

            j += sp.sum(beacc * bea - 2 * bea * dc) + sp.sum(d ** 2)
            k += sp.sum(beea * dc - beacc * beea) * 4 * delta

        return j / (nt * d.size), k / (nt * d.size)


    def _msge_with_gradient_overdetermined(self, data, delta, xvschema, skipstep):
        (l, m, t) = data.shape
        d = None
        l, k = 0, 0
        nt = sp.ceil(t / skipstep)
        for s in range(0, t, skipstep):
            #print(s,drange)
            trainset, testset = xvschema(s, t)

            (a, b) = self._construct_eqns(sp.atleast_3d(data[:, :, trainset]))
            (c, d) = self._construct_eqns(sp.atleast_3d(data[:, :, testset]))

            #e = sp.linalg.inv(np.eye(a.shape[1])*delta**2 + a.transpose().dot(a), overwrite_a=True, check_finite=False)
            e = sp.linalg.inv(sp.eye(a.shape[1]) * delta ** 2 + a.transpose().dot(a))

            ba = b.transpose().dot(a)
            dc = d.transpose().dot(c)
            bae = ba.dot(e)
            baee = bae.dot(e)
            baecc = bae.dot(c.transpose().dot(c))

            l += sp.sum(baecc * bae - 2 * bae * dc) + sp.sum(d ** 2)
            k += sp.sum(baee * dc - baecc * baee) * 4 * delta

        return l / (nt * d.size), k / (nt * d.size)

    def _get_msge_with_gradient_func(self, shape):
        (l, m, t) = shape

        n = (l - self.p) * t
        underdetermined = n < m * self.p

        if underdetermined:
            return self._msge_with_gradient_underdetermined
        else:
            return self._msge_with_gradient_overdetermined
