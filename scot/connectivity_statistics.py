# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013-2016 SCoT Development Team

"""Routines for statistical evaluation of connectivity."""

from __future__ import division

import numpy as np

from .datatools import randomize_phase, atleast_3d
from .connectivity import connectivity
from .utils import cartesian
from .parallel import parallel_loop


def surrogate_connectivity(measures, data, var, nfft=512, repeats=100,
                           n_jobs=1, verbose=0):
    """Calculate surrogate connectivity for a multivariate time series by phase
    randomization [1]_.

    .. note:: Parameter `var` will be modified by the function. Treat as
    undefined after the function returns.

    Parameters
    ----------
    measures : str or list of str
        Name(s) of the connectivity measure(s) to calculate. See
        :class:`Connectivity` for supported measures.
    data : array, shape (trials, channels, samples) or (channels, samples)
        Time series data (2D or 3D for multiple trials)
    var : VARBase-like object
        Instance of a VAR model.
    nfft : int, optional
        Number of frequency bins to calculate. Note that these points cover the
        range between 0 and half the sampling rate.
    repeats : int, optional
        Number of surrogate samples to take.
    n_jobs : int | None, optional
        Number of jobs to run in parallel. If set to None, joblib is not used
        at all. See `joblib.Parallel` for details.
    verbose : int, optional
        Verbosity level passed to joblib.

    Returns
    -------
    result : array, shape (`repeats`, n_channels, n_channels, nfft)
        Values of the connectivity measure for each surrogate. If
        `measure_names` is a list of strings a dictionary is returned, where
        each key is the name of the measure, and the corresponding values are
        arrays of shape (`repeats`, n_channels, n_channels, nfft).

    .. [1] J. Theiler et al. Testing for nonlinearity in time series: the
           method of surrogate data. Physica D, 58: 77-94, 1992.
    """
    par, func = parallel_loop(_calc_surrogate, n_jobs=n_jobs, verbose=verbose)
    output = par(func(randomize_phase(data), var, measures, nfft)
                 for _ in range(repeats))
    return convert_output_(output, measures)


def _calc_surrogate(data, var, measure_names, nfft):
    var.fit(data)
    return connectivity(measure_names, var.coef, var.rescov, nfft)


def jackknife_connectivity(measures, data, var, leaveout=1, nfft=512,
                           n_jobs=1, verbose=0):
    """Calculate jackknife estimates of connectivity.

    For each jackknife estimate a block of trials is left out. This is repeated
    until each trial was left out exactly once. The number of estimates depends
    on the number of trials and the value of `leaveout`. It is calculated by
    repeats = `n_trials` // `leaveout`.

    .. note:: Parameter `var` will be modified by the function. Treat as
    undefined after the function returns.

    Parameters
    ----------
    measures : str or list of str
        Name(s) of the connectivity measure(s) to calculate. See
        :class:`Connectivity` for supported measures.
    data : array, shape (trials, channels, samples)
        Time series data (multiple trials).
    var : VARBase-like object
        Instance of a VAR model.
    leaveout : int, optional
        Number of trials to leave out in each estimate.
    nfft : int, optional
        Number of frequency bins to calculate. Note that these points cover the
        range between 0 and half the sampling rate.
    leaveout : int, optional
        Number of trials to leave out in each estimate.
    n_jobs : int | None, optional
        Number of jobs to run in parallel. If set to None, joblib is not used
        at all. See `joblib.Parallel` for details.
    verbose : int, optional
        Verbosity level passed to joblib.

    Returns
    -------
    result : array, shape (`repeats`, n_channels, n_channels, nfft)
        Values of the connectivity measure for each surrogate. If
        `measure_names` is a list of strings a dictionary is returned, where
        each key is the name of the measure, and the corresponding values are
        arrays of shape (`repeats`, n_channels, n_channels, nfft).
    """
    data = atleast_3d(data)
    t, m, n = data.shape

    assert(t > 1)

    if leaveout < 1:
        leaveout = int(leaveout * t)

    num_blocks = t // leaveout

    mask = lambda block: [i for i in range(t) if i < block*leaveout or
                                                 i >= (block + 1) * leaveout]

    par, func = parallel_loop(_calc_jackknife, n_jobs=n_jobs, verbose=verbose)
    output = par(func(data[mask(b), :, :], var, measures, nfft)
                 for b in range(num_blocks))
    return convert_output_(output, measures)


def _calc_jackknife(data_used, var, measure_names, nfft):
    var.fit(data_used)
    return connectivity(measure_names, var.coef, var.rescov, nfft)


def bootstrap_connectivity(measures, data, var, num_samples=None, repeats=100,
                           nfft=512, n_jobs=1, verbose=0):
    """Calculate bootstrap estimates of connectivity.

    To obtain a bootstrap estimate trials are sampled randomly with replacement
    from the data set.

    .. note:: Parameter `var` will be modified by the function. Treat as
    undefined after the function returns.

    Parameters
    ----------
    measures : str or list of str
        Name(s) of the connectivity measure(s) to calculate. See
        :class:`Connectivity` for supported measures.
    data : array, shape (trials, channels, samples)
        Time series data (multiple trials).
    var : VARBase-like object
        Instance of a VAR model.
    num_samples : int, optional
        Number of samples to take for each bootstrap estimates. Defaults to the
        same number of trials as present in the data.
    repeats : int, optional
        Number of bootstrap estimates to take.
    nfft : int, optional
        Number of frequency bins to calculate. Note that these points cover the
        range between 0 and half the sampling rate.
    n_jobs : int | None, optional
        Number of jobs to run in parallel. If set to None, joblib is not used
        at all. See `joblib.Parallel` for details.
    verbose : int, optional
        Verbosity level passed to joblib.

    Returns
    -------
    measure : array, shape (`repeats`, n_channels, n_channels, nfft)
        Values of the connectivity measure for each bootstrap estimate. If
        `measure_names` is a list of strings a dictionary is returned, where
        each key is the name of the measure, and the corresponding values are
        arrays of shape (`repeats`, n_channels, n_channels, nfft).
    """
    data = atleast_3d(data)
    n, m, t = data.shape

    assert(t > 1)

    if num_samples is None:
        num_samples = t

    mask = lambda r: np.random.random_integers(0, data.shape[0]-1, num_samples)

    par, func = parallel_loop(_calc_bootstrap, n_jobs=n_jobs, verbose=verbose)
    output = par(func(data[mask(r), :, :], var, measures, nfft, num_samples)
                 for r in range(repeats))
    return convert_output_(output, measures)


def _calc_bootstrap(data, var, measures, nfft):
    var.fit(data)
    return connectivity(measures, var.coef, var.rescov, nfft)


def test_bootstrap_difference(a, b):
    """Test mean difference between two bootstrap estimates.

    This function calculates the probability p of observing a more extreme mean
    difference between `a` and `b` under the null hypothesis that `a` and `b`
    come from the same distribution.

    If p is smaller than e.g. 0.05, we can reject the null hypothesis at an
    alpha-level of 0.05 and conclude that `a` and `b` likely come from
    different distributions.

    .. note:: p-values are calculated along the first dimension. Thus,
              channels * channels * nfft individual p-values are obtained. To
              determine if a difference is significant, it is important to
              correct for multiple testing.

    Parameters
    ----------
    a, b : array, shape (`repeats`, channels, channels, nfft)
        Two bootstrap estimates to compare. The number of repetitions (first
        dimension) does not have to be equal.

    Returns
    -------
    p : array, shape (channels, channels, nfft)
        p-values.

    Notes
    -----
    The function estimates the distribution of `b[j]` - `a[i]` by calculating
    the difference for each combination of `i` and `j`. The total number of
    difference samples available is therefore a.shape[0] * b.shape[0]. The
    p-value is calculated as the smallest percentile of that distribution that
    does not contain 0.

    See also
    --------
    :func:`significance_fdr` : Correct for multiple testing by controlling the
    false discovery rate.
    """
    old_shape = a.shape[1:]
    a = np.asarray(a).reshape((a.shape[0], -1))
    b = np.asarray(b).reshape((b.shape[0], -1))

    n = a.shape[0]

    s1, s2 = 0, 0
    for i in cartesian((np.arange(n), np.arange(n))):
        c = b[i[1], :] - a[i[0], :]
        s1 += c >= 0
        s2 += c <= 0

    p = np.minimum(s1, s2) / (n*n)
    return p.reshape(old_shape)


def significance_fdr(p, alpha):
    """Calculate significance by controlling for the false discovery rate.

    This function determines which of the p-values in `p` can be considered
    significant. Correction for multiple comparisons is performed by
    controlling the false discovery rate (FDR). The FDR is the maximum fraction
    of p-values that are wrongly considered significant [1]_.

    Parameters
    ----------
    p : array, shape (channels, channels, nfft)
        p-values.
    alpha : float
        Maximum false discovery rate.

    Returns
    -------
    s : array, dtype=bool, shape (channels, channels, nfft)
        Significance of each p-value.

    References
    ----------
    .. [1] Y. Benjamini, Y. Hochberg. Controlling the false discovery rate: a
           practical and powerful approach to multiple testing. J. Royal Stat.
           Soc. Series B 57(1): 289-300, 1995.
    """
    i = np.argsort(p, axis=None)
    m = i.size - np.sum(np.isnan(p))

    j = np.empty(p.shape, int)
    j.flat[i] = np.arange(1, i.size + 1)

    mask = p <= alpha * j / m

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
        output = dict((m, np.array([output[r][m] for r in range(repeats)]))
                      for m in measures)
        return output
