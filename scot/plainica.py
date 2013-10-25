# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

from . import config
from .datatools import cat_trials, dot_special
from . import xvschema
from . import var

import numpy as np


def plainica(x, reducedim=0.99, backend=None):
    '''
    plainica( x )
    plainica( x, reducedim, backend )
    
    Apply ICA to the data x, with optional PCA dimensionality reduction.
    
    Parameters     Default  Shape   Description
    --------------------------------------------------------------------------
    x              :      : n,m,t : 3d data matrix (n samples, m signals, t trials)
                          : n,m   : 2d data matrix (n samples, m signals)
    reducedim      :      : 0.99  : a number less than 1 is interpreted as the
                                    fraction of variance that should remain in
                                    the data. All components that describe in
                                    total less than 1-retain_variance of the
                                    variance in the data are removed by the PCA.
                                    An integer number of 1 or greater is
                                    interpreted as the number of components to
                                    keep after applying the PCA.
                                    If set to 'no_pca' the PCA step is skipped.
    backend        : None :       : backend to use for processing (see backend
                                    module for details). If backend==None, the
                                    backend set in config will be used.
    
    Output
    --------------------------------------------------------------------------
    U   Unmixing matrix
    m   Mixing matrix
    '''

    x = np.atleast_3d(x)
    l, m, t = np.shape(x)

    if backend is None:
        backend = config.backend

    # pre-transform the data with PCA
    if reducedim == 'no pca':
        c = np.eye(m)
        d = np.eye(m)
        xpca = x
    else:
        c, d, xpca = backend['pca'](x, reducedim)
        m = c.shape[1]

    # run on residuals ICA to estimate volume conduction    
    mx, ux = backend['ica'](cat_trials(xpca))

    # correct (un)mixing matrix estimatees
    mx = mx.dot(d)
    ux = c.dot(ux)

    class Result:
        unmixing = ux
        mixing = mx

    return Result
