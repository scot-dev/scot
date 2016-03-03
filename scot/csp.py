# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013-2016 SCoT Development Team

"""Common spatial patterns (CSP) implementation."""

import numpy as np
from scipy.linalg import eigh
    

def csp(x, cl, numcomp=None):
    """Calculate common spatial patterns (CSP).

    Parameters
    ----------
    x : array, shape (trials, channels, samples) or (channels, samples)
        EEG data set.
    cl : list of valid dict keys
        Class labels associated with each trial. Currently, only two classes
        are supported.
    numcomp : int, optional
        Number of patterns to keep after applying CSP. If `numcomp` is greater
        than channels or None, all patterns are returned.

    Returns
    -------
    w : array, shape (channels, components)
        CSP weight matrix.
    v : array, shape (components, channels)
        CSP projection matrix.
    """

    x = np.asarray(x)
    cl = np.asarray(cl).ravel()

    if x.ndim != 3 or x.shape[0] < 2:
        raise AttributeError('CSP requires at least two trials.')

    t, m, n = x.shape
    
    if t != cl.size:
        raise AttributeError('CSP only works with multiple classes. Number of '
                             'elements in cl ({}) must equal the first '
                             'dimension of x ({})'.format(cl.size, t))

    labels = np.unique(cl)
    
    if labels.size != 2:
        raise AttributeError('CSP is currently implemented for two classes '
                             'only (got {}).'.format(labels.size))
        
    x1 = x[cl == labels[0], :, :]
    x2 = x[cl == labels[1], :, :]
    
    sigma1 = np.zeros((m, m))
    for t in range(x1.shape[0]):
        sigma1 += np.cov(x1[t, :, :]) / x1.shape[0]
    sigma1 /= sigma1.trace()
    
    sigma2 = np.zeros((m, m))
    for t in range(x2.shape[0]):
        sigma2 += np.cov(x2[t, :, :]) / x2.shape[0]
    sigma2 /= sigma2.trace()

    e, w = eigh(sigma1, sigma1 + sigma2, overwrite_a=True, overwrite_b=True,
                check_finite=False)

    order = np.argsort(e)[::-1]
    w = w[:, order]
    v = np.linalg.inv(w)
   
    # subsequently remove unwanted components from the middle of w and v
    if numcomp is None:
        numcomp = w.shape[1]
    while w.shape[1] > numcomp:
        i = int(np.floor(w.shape[1]/2))
        w = np.delete(w, i, 1)
        v = np.delete(v, i, 0)
        
    return w, v
