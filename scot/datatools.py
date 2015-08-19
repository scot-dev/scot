# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2015 SCoT Development Team

"""
Summary
-------
Tools for basic data manipulation.
"""

import numpy as np


def cut_segments(x2d, tr, start, stop):
    """Cut continuous signal into segments.

    This function cuts segments from a continuous signal.

    Parameters
    ----------
    data : array, shape (m, n)
        Input data with m signals and n samples.
    tr : list of int
        Trigger positions.
    start : int
        Window start (offset relative to trigger).
    stop : int
        Window end (offset relative to trigger).

    Returns
    -------
    x3d : array, shape (len(tr), m, stop-start)
        Segments cut from data. Individual segments are stacked along the first
        dimension.

    See also
    --------
    cat_trials : Concatenate segments.

    Examples
    --------
    >>> data = np.random.randn(5, 1000)  # 5 channels, 1000 samples
    >>> tr = [750, 500, 250]  # three segments
    >>> x3d = cut_segments(data, tr, 50, 100)  # each segment is 50 samples
    >>> x3d.shape
    (3, 5, 50)
    """
    x2d = np.atleast_2d(x2d)
    segment = np.arange(start, stop)
    return np.concatenate([x2d[np.newaxis, :, t + segment] for t in tr])


def cat_trials(x3d):
    """Concatenate trials along time axis.

    Parameters
    ----------
    x3d : array, shape (t, m, n)
        Segmented input data with t trials, m signals, and n samples.

    Returns
    -------
    x2d : array, shape (m, t * n)
        Trials are concatenated along the second axis.

    See also
    --------
    cut_segments : Cut segments from continuous data.

    Examples
    --------
    >>> x = np.random.randn(6, 4, 150)
    >>> y = cat_trials(x)
    >>> y.shape
    (4, 900)
    """
    x3d = assert_3d(x3d)
    t = x3d.shape[0]
    return np.concatenate(np.split(x3d, t, 0), axis=2).squeeze()


def dot_special(x2d, x3d):
    """Segment-wise dot product.

    This function calculates the dot product of x2d with each trial of x3d.

    Parameters
    ----------
    x2d : array, shape (p, m)
        Input argument.
    x3d : array, shape (t, m, n)
        Segmented input data with t trials, m signals, and n samples. The dot
        product with x2d is calculated for each trial.

    Returns
    -------
    out : array, shape (t, p, n)
        Dot product of x2d with each trial of x3d.

    Examples
    --------
    >>> x = np.random.randn(150, 40, 6)
    >>> a = np.ones((7, 40))
    >>> y = dot_special(a, x)
    >>> y.shape
    (150, 7, 6)
    """
    x3d = assert_3d(x3d)
    x2d = np.atleast_2d(x2d)
    return np.concatenate([x2d.dot(x3d[i, ...])[np.newaxis, ...]
                           for i in range(x3d.shape[0])])


def randomize_phase(data):
    """Phase randomization.

    This function randomizes the spectral phase of the input data along the
    first dimension.

    Parameters
    ----------
    data : array
        Input array.

    Returns
    -------
    out : array
        Array of same shape as data.

    Notes
    -----
    The algorithm randomizes the phase component of the input's complex Fourier
    transform.

    Examples
    --------
    .. plot::
        :include-source:

        from pylab import *
        from scot.datatools import randomize_phase
        np.random.seed(1234)
        s = np.sin(np.linspace(0,10*np.pi,1000)).T
        x = np.vstack([s, np.sign(s)]).T
        y = randomize_phase(x)
        subplot(2,1,1)
        title('Phase randomization of sine wave and rectangular function')
        plot(x), axis([0,1000,-3,3])
        subplot(2,1,2)
        plot(y), axis([0,1000,-3,3])
        plt.show()
    """
    data = np.asarray(data)
    data_freq = np.fft.rfft(data, axis=0)
    data_freq = np.abs(data_freq) * np.exp(1j*np.random.random_sample(data_freq.shape)*2*np.pi)
    return np.fft.irfft(data_freq, data.shape[0], axis=0)


def assert_3d(x):
    if x.ndim >= 3:
        return x
    elif x.ndim == 2:
        return x[np.newaxis, ...]
    else:
        return x[np.newaxis, np.newaxis, :]
