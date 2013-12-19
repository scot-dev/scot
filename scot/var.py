# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

""" vector autoregressive (VAR) model """

import numbers
from functools import partial

import numpy as np
import scipy as sp

from . import datatools
from . import xvschema as xv
from .utils import acm
from .datatools import cat_trials
from .utils import DocStringInheritor


class Defaults:
    xvschema = xv.multitrial


class VARBase(DocStringInheritor):
    """ Represents a vector autoregressive (VAR) model.

        Note on the arrangement of model coefficients:
            b is of shape m, m*p, with sub matrices arranged as follows:
                b_00 b_01 ... b_0m
                b_10 b_11 ... b_1m
                .... ....     ....
                b_m0 b_m1 ... b_mm
            Each sub matrix b_ij is a column vector of length p that contains the
            filter coefficients from channel j (source) to channel i (sink).
    """

    def __init__(self, model_order):
        """ Create a new VAR model instance.

            Parameters     Default  Shape   Description
            --------------------------------------------------------------------------
            model_order    :      :       : Autoregressive model order
        """
        self.p = model_order
        self.coef = None
        self.residuals = None
        self.rescov = None

    def copy(self):
        other = self.__class__(self.p)
        other.coef = self.coef.copy()
        other.residuals = self.residuals.copy()
        other.rescov = self.rescov.copy()
        return other

    def fit(self, data):
        """ Fit the model to data.

            Parameters     Default  Shape   Description
            --------------------------------------------------------------------------
            data             :      : N,M,T : 3d data matrix (N samples, M signals, T trials)
                             :      : N,M   : 2d data matrix (N samples, M signals)
        """
        raise NotImplementedError('method fit() is not implemented in ' + str(self))

    def optimize(self, data):
        """ Optimize the var model's hyperparameters (such as regularization).

            Parameters     Default  Shape   Description
            --------------------------------------------------------------------------
            data             :      : N,M,T : 3d data matrix (N samples, M signals, T trials)
                             :      : N,M   : 2d data matrix (N samples, M signals)
        """
        raise NotImplementedError('method optimize() is not implemented in ' + str(self))
        return self

    def simulate(self, l, noisefunc=None):
        """ Simulate vector autoregressive (VAR) model with optional noise generating function.

            Parameters     Default  Shape   Description
            --------------------------------------------------------------------------
            l              :      : 1     : Number of samples to generate
                           :      : 2     : l[0]: number of samples, l[1]: number of trials
            noisefunc      : None :       : callback function that takes no parameter and returns m values
                                            This function is used to create the generating
                                            noise process. if noisefunc==None Gaussian
                                            white noise with zero mean and unit variance
                                            is created.

            Output           Shape   Description
            --------------------------------------------------------------------------
            data           : L,M,T  : 3D data matrix
        """
        (m, n) = sp.shape(self.coef)
        p = n // m

        try:
            (l, t) = l
        except TypeError:
            t = 1

        if noisefunc is None:
            noisefunc = lambda: sp.random.normal(size=(1, m))

        n = l + 10 * p

        y = sp.zeros((n, m, t))
        res = sp.zeros((n, m, t))

        for s in range(t):
            for i in range(p):
                e = noisefunc()
                res[i, :, s] = e
                y[i, :, s] = e
            for i in range(p, n):
                e = noisefunc()
                res[i, :, s] = e
                y[i, :, s] = e
                for k in range(1, p + 1):
                    y[i, :, s] += self.coef[:, (k - 1)::p].dot(y[i - k, :, s])

        self.residuals = res[10 * p:, :, :]
        self.rescov = sp.cov(cat_trials(self.residuals), rowvar=False)

        return y[10 * p:, :, :]

    def predict(self, data):
        """ Predict samples from actual data.

            Note that the model requires p past samples for prediction. Thus, the first
            p samples are invalid and set to 0, where p is the model order.

            Parameters     Default  Shape   Description
            --------------------------------------------------------------------------
            data             :      : N,M,T : 3d data matrix (N samples, M signals, T trials)
                             :      : N,M   : 2d data matrix (N samples, M signals)

            Output           Shape   Description
            --------------------------------------------------------------------------
            predicted      : L,M,T : 3D data matrix
        """
        data = sp.atleast_3d(data)
        (l, m, t) = data.shape

        p = int(sp.shape(self.coef)[1] / m)

        y = sp.zeros(data.shape)
        for k in range(1, p + 1):
            bp = self.coef[:, (k - 1)::p]
            for n in range(p, l):
                y[n, :, :] += bp.dot(data[n - k, :, :])
        return y

    def is_stable(self):
        """ Test if the VAR model is stable.

            Output           Shape   Description
            --------------------------------------------------------------------------
            stable         :       : True of False if the model is stable/unstable

            References:
            [1] H. Lütkepohl, "New Introduction to Multiple Time Series Analysis", 2005, Springer, Berlin, Germany
        """
        m, mp = self.coef.shape
        p = mp // m
        assert(mp == m*p)

        top_block = []
        for i in range(p):
            top_block.append(self.coef[:, i::p])
        top_block = np.hstack(top_block)

        im = np.eye(m)
        eye_block = im
        for i in range(p-2):
            eye_block = sp.linalg.block_diag(im, eye_block)
        eye_block = np.hstack([eye_block, np.zeros((m*(p-1), m))])

        tmp = np.vstack([top_block, eye_block])

        return np.all(np.abs(np.linalg.eig(tmp)[0]) < 1)

    def test_whiteness(self, h, repeats=100, get_q=False):
        """ Test if the VAR model residuals are white (uncorrelated up to a lag of h).

            This function calculates the Li-McLeod as Portmanteau test statistic Q to
            test against the null hypothesis H0: "the residuals are white" [1].
            Surrogate data for H0 is created by sampling from random permutations of
            the residuals.

            Usually the returned p-value is compared against a pre-defined type 1 error
            level of alpha=0.05 or alpha=0.01. If p<=alpha, the hypothesis of white
            residuals is rejected, which indicates that the VAR model does not properly
            describe the data.

            Parameters     Default  Shape   Description
            --------------------------------------------------------------------------
            h              :      :       : The test is performed for all time lags up
                                            to h. Note that according to [2] h must
                                            satisfy h = O(n^0.5), where n is the length
                                            (time samples) of the residuals.
            repeats        : 100  :       : Number of samples to create under the null
                                            hypothesis. Larger number will give more
                                            accurate results.
            get_q          : False:       : If set to False only the p-value is returned.
                                            Otherwise actual values of the Li-McLeod
                                            statistic are returned too.

            Output           Shape   Description
            --------------------------------------------------------------------------
            pr               :       : Probability of observing a more extreme value of
                                       Q under the assumption that H0 is true.
            q0               :       : (optional, see get_q) list of values that created
                                       as surrogates to estimate the distribution of Q
                                       under the null-hypothesis.
            q                :       : (optional, see get_q) Value of Q that corresponds
                                       to the current residuals.

            References:
            [1] H. Lütkepohl, "New Introduction to Multiple Time Series Analysis", 2005, Springer, Berlin, Germany
            [2] J.R.M. Hosking, "The Multivariate Portmanteau Statistic", 1980, J. Am. Statist. Assoc.
        """
        res = self.residuals[self.p:, :, :]
        (n, m, t) = res.shape
        nt = (n-self.p)*t

        q0 = _calc_q_h0(repeats, res, h, nt)[:,2,-1]
        q = _calc_q_statistic(res, h, nt)[2,-1]

        # probability of observing a result more extreme than q under the null-hypothesis
        pr = np.sum(q0 >= q) / repeats

        if get_q:
            return pr, q0, q
        else:
            return pr

    def _construct_eqns(self, data):
        """Construct VAR equation system"""
        (l, m, t) = np.shape(data)
        n = (l - self.p) * t     # number of linear relations
        # Construct matrix x (predictor variables)
        x = np.zeros((n, m * self.p))
        for i in range(m):
            for k in range(1, self.p + 1):
                x[:, i * self.p + k - 1] = np.reshape(data[self.p - k:-k, i, :], n)

        # Construct vectors yi (response variables for each channel i)
        y = np.zeros((n, m))
        for i in range(m):
            y[:, i] = np.reshape(data[self.p:, i, :], n)

        return x, y


def fit_multiclass(data, cl, p, delta=None, return_residuals=False, return_covariance=False):
    """
    fit_multiclass( data, cl, p )
    fit_multiclass( data, cl, p, delta )

    Fits a separate autoregressive model for each class.

    If sqrtdelta is provited and nonzero, the least squares estimation is
    regularized with ridge regression.

    Parameters     Default  Shape   Description
    --------------------------------------------------------------------------
    data             :      : n,m,T : 3d data matrix (n samples, m signals, T trials)
    cl               :      : T     : class label for each trial
    p                :      :       : Model order can be scalar (same model order for each class)
                                      or a dictionary that contains a key for each unique class label
                                      to specify the model order for each class individually.
    delta            : None :       : regularization parameter
    return_residuals : False :      : if True, also return model residuals
    return_covariance: False :      : if True, also return dictionary of covariances

    Output
    --------------------------------------------------------------------------
    bcl   dictionary of model coefficients for each class
    res   (optional) Model residuals: (same shape as data), note that
          the first p (depending on the class) residuals are invalid.
    ccl   (optional) dictionary of residual covariances for each class
    """

    data = np.atleast_3d(data)
    cl = np.asarray(cl)

    labels = np.unique(cl)

    if cl.size != data.shape[2]:
        raise AttributeError(
            'cl must contain a class label for each trial (expected size %d, but got %d).' % (data.shape[2], cl.size))

    if isinstance(p, numbers.Number):
        p = dict.fromkeys(labels, p)
    else:
        try:
            assert (set(labels) == set(p.keys()))
        except:
            raise AttributeError(
                'Model order p must be either a scalar number, or a dictionary containing a key for each unique label in cl.')

    bcl, ccl = {}, {}
    res = np.zeros(data.shape)
    for c in labels:
        x = data[:, :, cl == c]
        b = fit(x, p[c], delta)
        bcl[c] = b

        if return_residuals or return_covariance:
            r = x - predict(x, b)

        if return_residuals:
            res[:, :, cl == c] = r

        if return_covariance:
            ccl[c] = np.cov(datatools.cat_trials(r), rowvar=False)

    result = []

    if return_residuals or return_covariance:
        result.append(bcl)
    else:
        return bcl

    if return_residuals:
        result.append(res)

    if return_covariance:
        result.append(ccl)

    return tuple(result)


############################################################################


def _calc_q_statistic(x, h, nt):
    """ calculate portmanteau statistics up to a lag of h.
    """
    (n, m, t) = x.shape

    # covariance matrix of x
    c0 = acm(x, 0)

    # LU factorization of covariance matrix
    c0f = sp.linalg.lu_factor(c0, overwrite_a=False, check_finite=True)

    q = np.zeros((3, h+1))
    for l in range(1, h+1):
        cl = acm(x, l)

        # calculate tr(cl' * c0^-1 * cl * c0^-1)
        a = sp.linalg.lu_solve(c0f, cl)
        b = sp.linalg.lu_solve(c0f, cl.T)
        tmp = a.dot(b).trace()

        # Box-Pierce
        q[0, l] = tmp

        # Ljung-Box
        q[1, l] = tmp / (nt-l)

        # Li-McLeod
        q[2, l] = tmp

    q *= nt
    q[1, :] *= (nt+2)

    q = np.cumsum(q, axis=1)

    for l in range(1, h+1):
        q[2, l] = q[0, l] + m*m*l*(l+1) / (2*nt)

    return q


def _calc_q_h0(n, x, h, nt):
    """ calculate q under the null-hypothesis of whiteness
    """
    x = x.copy()

    q = []
    for i in range(n):
        np.random.shuffle(x)    # shuffle along time axis
        q.append(_calc_q_statistic(x, h, nt))
    return np.array(q)
