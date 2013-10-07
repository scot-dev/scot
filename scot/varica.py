# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

from . import config
from .datatools import cat_trials, dot_special
from . import xvschema
from . import var

import numpy as np

def mvarica(X, P, retain_variance=0.99, numcomp=None, delta=0):
    '''
    mvarica( X, P )
    mvarica( X, P, retain_variance, delta )
    mvarica( X, P, numcomp, delta )
    
    Apply MVARICA to the data X. MVARICA performs the following steps:
        1. Optional dimensionality reduction with PCA
        2. Fitting a VAR model tho the data
        3. Decomposing the VAR model residuals with ICA
        4. Correcting the VAR coefficients
    
    Parameters     Default  Shape   Description
    --------------------------------------------------------------------------
    X              :      : N,M,T : 3d data matrix (N samples, M signals, T trials)
                          : N,M   : 2d data matrix (N samples, M signals)
    P              :      :       : VAR model order
    retain_variance:      : 0.99  : If specified as a number it is interpreted
                                    as the fraction of variance that should
                                    remain in the data. All components that
                                    describe in total less than 1-retain_variance
                                    of the variance in the data are removed by
                                    the PCA.
                                    If set to 'no_pca' the PCA step is skipped.
    numcomp        : None :       : Can be provided instead of retain_variance
                                    to specify the exact number of components
                                    to keep. The PCA keeps numcomp components
                                    with the highest variance and discards the
                                    rest.
    delta          : 0    :       : regularization parameter for VAR fitting
                                    set to 'auto' to determine optimal setting
    
    Output
    --------------------------------------------------------------------------
    B   Model coefficients: [B_0, B_1, ... B_P], each sub matrix B_k is of size M*M
    U   Unmixing matrix
    M   Mixing matrix
    e   Residual process
    C   Residual covariance matrix
    delta   Regularization parameter
    
    Note on the arrangement of model coefficients:
        B is of shape M, M*P, with sub matrices arranged as follows:
            b_00 b_01 ... b_0M
            b_10 b_11 ... b_1M
            .... ....     ....
            b_M0 b_M1 ... b_MM
        Each sub matrix b_ij is a column vector of length P that contains the
        filter coefficients from channel j (source) to channel i (sink).
    '''
    
    X = np.atleast_3d(X)
    L, M, T = np.shape(X)
    
    # pre-transform the data with PCA
    if retain_variance == 'no pca':
        C = np.eye(M)
        D = np.eye(M)
        Xpca = X
    else:
        C, D, Xpca = config.backend['pca'](X, retain_variance, numcomp)
        M = C.shape[1]
    
    if delta == 'auto':
        delta = var.optimize_delta_bisection( Xpca[:,:,:], P, xvschema=xvschema.multitrial )
    
    # fit MVAR model
    A = var.fit( Xpca, P, delta )
    
    # residuals
    r = Xpca - var.predict( Xpca, A )

    # run on residuals ICA to estimate volume conduction    
    Mx, Ux = config.backend['ica'](cat_trials(r))
    
    # driving process
    e = dot_special(r, Ux)

    # correct AR coefficients
    B = np.zeros(A.shape)
    for p in range(0,P):
        B[:,p::P] = Mx.dot(A[:,p::P].transpose()).dot(Ux).transpose()
    
    # correct (un)mixing matrix estimatees
    Mx = Mx.dot(D)
    Ux = C.dot(Ux)
    
    class result:
        unmixing = Ux
        mixing = Mx
        residuals = e
        C = np.cov(cat_trials(e), rowvar=False)
    result.delta = delta
    result.B = B
        
    
    return result
