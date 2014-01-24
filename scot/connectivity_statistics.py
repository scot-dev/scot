# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013-2014 SCoT Development Team

import numpy as np
import scipy as sp
from .datatools import randomize_phase
from .connectivity import connectivity
from . import config


def surrogate_connectivity(measure_names, data, var, nfft=512, repeats=100):
    """ Calculates surrogate connectivity for a multivariate time series by phase randomization [1].

    .. note:: Parameter `var` will be modified by the function. Treat as undefined after the function returned.

    Parameters
    ----------
    measure_names : {str, list of str}
        Name(s) of the connectivity measure(s) to calculate. See :class:`Connectivity` for supported measures.
    data : ndarray, shape = [n_samples, n_channels, (n_trials)]
        Time series data (2D or 3D for multiple trials)
    var : VARBase-like object
        Instance of a VAR model.


        Parameters     Default  Shape   Description
        --------------------------------------------------------------------------
        measures       :      :       : String or list of strings. Each string is
                                        the (case sensitive) name of a connectivity
                                        measure to calculate. See documentation of
                                        Connectivity for supported measures.
                                        The function returns an ndarray if measures
                                        is a string, otherwise a dict is returned.
        data           :      : n,m,t : 3d data matrix (n samples, m signals, t trials)
                              : n,m   : 2d data matrix (n samples, m signals)
        var            :      :       : Instance of class that represents VAR models.
        nfft           : 512  : 1     : Number of frequency bins to calculate. Note
                                        that these points cover the range between 0
                                        and the nyquist frequency.
        repeats        : 100  : 1     : Number of surrogates to create.

        Output   Shape               Description
        --------------------------------------------------------------------------
        result : repeats, m,m,nfft : An ndarray of shape (repeats, m, m, nfft) is
                                     returned if measures is a string. If measures
                                     is a list of strings a dictionary is returned,
                                     where each key is the name of the measure, and
                                     the corresponding values are ndarrays of shape
                                     (repeats, m, m, nfft).

        [1] J. Theiler et al. "Testing for nonlinearity in time series: the method of surrogate data", Physica D,
            vol 58, pp. 77-94, 1992
    """
    output = []
    for r in range(repeats):
        surrogate_data = randomize_phase(data)
        var.fit(surrogate_data)
        c = connectivity(measure_names, var.coef, var.rescov, nfft)
        output.append(c)
    return convert_output_(output, measure_names)


def jackknife_connectivity(measures, data, var, nfft=512, leaveout=1):
    """ Calculates Jackknife estimates of connectivity by leaving out trials.

        Parameters     Default  Shape   Description
        --------------------------------------------------------------------------
        measures       :      :       : String or list of strings. Each string is
                                        the (case sensitive) name of a connectivity
                                        measure to calculate. See documentation of
                                        Connectivity for supported measures.
                                        The function returns an ndarray if measures
                                        is a string, otherwise a dict is returned.
        data           :      : n,m,t : 3d data matrix (n samples, m signals, t trials)
        var            :      :       : Instance of class that represents VAR models.
        nfft           : 512  : 1     : Number of frequency bins to calculate. Note
                                        that these points cover the range between 0
                                        and the nyquist frequency.
        leaveout       : 1    : 1     : Number of trials to leave out in each estimate.

        Output          Shape        Description
        --------------------------------------------------------------------------
        result        : r,m,m,nfft : An ndarray of shape (r, m, m, nfft) is returned
                                     returned (where r is t//leaveout) if measures
                                     is a string. If measures is a list of strings
                                     a dictionary is returned, where each key is the
                                     name of the measure, and the corresponding
                                     values are ndarrays of shape
                                     (repeats, m, m, nfft).
    """
    data = np.atleast_3d(data)
    n, m, t = data.shape

    if leaveout < 1:
        leaveout = int(leaveout * t)

    num_blocks = int(t / leaveout)

    output = []
    for b in range(num_blocks):
        mask = [i for i in range(t) if i < b*leaveout or i >= (b+1)*leaveout]
        data_used = data[:, :, mask]
        var.fit(data_used)
        c = connectivity(measures, var.coef, var.rescov, nfft)
        output.append(c)
    return convert_output_(output, measures)


def bootstrap_connectivity(measures, data, var, nfft=512, repeats=100, num_samples=None):
    """ Calculates Bootstrap estimates of connectivity by randomly sampling trials with replacement.

        Parameters     Default  Shape   Description
        --------------------------------------------------------------------------
        measures       :      :       : String or list of strings. Each string is
                                        the (case sensitive) name of a connectivity
                                        measure to calculate. See documentation of
                                        Connectivity for supported measures.
                                        The function returns an ndarray if measures
                                        is a string, otherwise a dict is returned.
        data           :      : n,m,t : 3d data matrix (n samples, m signals, t trials)
                              : n,m   : 2d data matrix (n samples, m signals)
        var            :      :       : Instance of class that represents VAR models.
        nfft           : 512  : 1     : Number of frequency bins to calculate. Note
                                        that these points cover the range between 0
                                        and the nyquist frequency.
        num_samples    : None : 1     : Number of trials to sample for each estimate. Default: t
        repeats        : 100  : 1     : Number of bootstrap estimates to calculate

        Output   Shape               Description
        --------------------------------------------------------------------------
        result : repeats, m,m,nfft : An ndarray of shape (repeats, m, m, nfft) is
                                     returned if measures is a string. If measures
                                     is a list of strings a dictionary is returned,
                                     where each key is the name of the measure, and
                                     the corresponding values are ndarrays of shape
                                     (repeats, m, m, nfft).
    """
    data = np.atleast_3d(data)
    n, m, t = data.shape

    if num_samples is None:
        num_samples = t

    output = []
    for r in range(repeats):
        mask = np.random.random_integers(0, t-1, num_samples)
        data_used = data[:, :, mask]
        var.fit(data_used)
        c = connectivity(measures, var.coef, var.rescov, nfft)
        output.append(c)
    return convert_output_(output, measures)


def test_bootstrap_difference(a, b):
    old_shape = a.shape[1:]
    a = np.asarray(a).reshape((a.shape[0], -1))
    b = np.asarray(b).reshape((b.shape[0], -1))

    n = a.shape[0]

    s1, s2 = 0, 0
    for i in config.backend['utils'].cartesian((np.arange(n), np.arange(n))):
        c = b[i[1], :] - a[i[0], :]

        s1 += c >= 0
        s2 += c <= 0

    p = np.minimum(s1, s2) / (n*n)

    return p.reshape(old_shape)


def test_rank_difference_a(a, b):
    """ Test for difference between two statistics with Mann-Whitney-U test.
        Samples along first dimension. p-values returned.
    """
    old_shape = a.shape[1:]
    assert(b.shape[1:] == old_shape)
    a = np.asarray(a).reshape((a.shape[0], -1))
    b = np.asarray(b).reshape((b.shape[0], -1))

    p = np.zeros(a.shape[1])

    for i in range(a.shape[1]):
        #u, pr = sp.stats.mannwhitneyu(a[:,i], b[:,i])
        t, pr = sp.stats.ttest_ind(a[:,i], b[:,i], equal_var=False)
        p[i] = pr

    return p.reshape(old_shape)


def significance_fdr(p, alpha):
    """ Get significance by controlling for the False Discovery Rate (FDR).
        Implemented the Benjamini-Hochberg procedure [1].

        [1] Y. Benjamini, Y. Hochberg, "Controlling the false discovery rate: a practical and powerful approach to
            multiple testing", Journal of the Royal Statistical Society, Series B 57(1), pp 289-300, 1995
    """
    i = np.argsort(p, axis=None)
    m = i.size - np.sum(np.isnan(p))

    j = np.empty(p.shape, int)
    j.flat[i] = np.arange(1, i.size+1)

    mask = p <= alpha*j/m

    if np.sum(mask) == 0:
        return mask

    # find largest k so that p_k <= alpha*k/m
    k = np.max(j[mask])

    # reject all H_i for i = 0...k
    s = j <= k

    return s


def convert_output_(output, measures):
    if isinstance(measures, str):
        return np.array(output)
    else:
        repeats = len(output)
        output = {m: np.array([output[r][m] for r in range(repeats)]) for m in measures}
        return output