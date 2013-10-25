# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

"""tools for basic data manipulation"""

import numpy as np


def cut_segments(rawdata, tr, start, stop):
    """
    x = cut_segments(rawdata, tr, start, stop):

    Cut continuous signal into segments of length L = stop - start.

    Parameters     Default  Shape   Description
    --------------------------------------------------------------------------
    rawdata        :      : n,m   : continuous signal data with n samples and m signals
    tr             :      : T     : list of trigger positions
    start          :      : 1     : window start (offset relative to trigger)
    stop           :      : 1     : window end (offset relative to trigger)

    Output
    --------------------------------------------------------------------------
    x              :      : L,m,T : 3d data matrix (L = stop - start)
    """
    rawdata = np.atleast_2d(rawdata)
    tr = np.array(tr, dtype='int').ravel()
    win = range(start, stop)
    return np.dstack([rawdata[tr[t] + win, :] for t in range(len(tr))])


def cat_trials(x):
    """
    y = cat_trials(x):

    Concatenate trials along time axis.

    Parameters     Default  Shape   Description
    --------------------------------------------------------------------------
    x              :      : l,m,t : 3d data matrix (L samples, m signals, t trials)

    Output
    --------------------------------------------------------------------------
    y              :      : l*t,m : 2d data matrix with trials concatenated along time axis
    """
    x = np.atleast_3d(x)
    t = x.shape[2]
    return np.squeeze(np.vstack(np.dsplit(x, t)), axis=2)


def dot_special(x, a):
    """
    y = dot_special(x, a):

    Dot product of 3D matrix x with 2D matrix a.

    This is equivalent to writing
        y = np.dstack([x[:,:,i].dot(a) for i in range(x.shape[2])])

    Computes y[:,:,i] = x[:,:,i].dot(a) for every i.

    Parameters     Default  Shape   Description
    --------------------------------------------------------------------------
    x              :      : L,m,T : 3d data matrix (L samples, m signals, T trials)
    a              :      : m,n   : 2d matrix to transform x with

    Output
    --------------------------------------------------------------------------
    y              :      : L,n,T : 3d data matrix y[:,:,i] = x[:,:,i].dot(a)
    """
    x = np.atleast_3d(x)
    a = np.atleast_2d(a)
    return np.dstack([x[:, :, i].dot(a) for i in range(x.shape[2])])
    