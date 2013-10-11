# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

from . import config
from .datatools import cat_trials, dot_special
from . import xvschema
from . import var

import numpy as np

def plainica(X, reducedim=0.99, backend=None):
    '''
    mvarica( X )
    mvarica( X, reducedim, backend )
    
    Apply ICA to the data X, with optional PCA dimensionality reduction.
    
    Parameters     Default  Shape   Description
    --------------------------------------------------------------------------
    X              :      : N,M,T : 3d data matrix (N samples, M signals, T trials)
                          : N,M   : 2d data matrix (N samples, M signals)
    reducedim      :      : 0.99  : A number less than 1 is interpreted as the
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
    M   Mixing matrix
    '''
    
    X = np.atleast_3d(X)
    L, M, T = np.shape(X)
    
    if backend == None:
        backend = config.backend
    
    # pre-transform the data with PCA
    if reducedim == 'no pca':
        C = np.eye(M)
        D = np.eye(M)
        Xpca = X
    else:
        C, D, Xpca = backend['pca'](X, reducedim)
        M = C.shape[1]

    # run on residuals ICA to estimate volume conduction    
    Mx, Ux = backend['ica'](cat_trials(Xpca))
    
    # correct (un)mixing matrix estimatees
    Mx = Mx.dot(D)
    Ux = C.dot(Ux)
    
    class result:
        unmixing = Ux
        mixing = Mx
        
    return result
