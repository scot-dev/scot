# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

from . import config
from .datatools import cat_trials, dot_special
from . import xvschema
from . import var

import numpy as np

def mvarica(X, P, reducedim=0.99, delta=0, backend=None):
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
    reducedim      : 0.99 :       : A number less than 1 is interpreted as the
                                    fraction of variance that should remain in
                                    the data. All components that describe in
                                    total less than 1-retain_variance of the
                                    variance in the data are removed by the PCA.
                                    An integer number of 1 or greater is
                                    interpreted as the number of components to
                                    keep after applying the PCA.
                                    If set to 'no_pca' the PCA step is skipped.
    delta          : 0    :       : regularization parameter for VAR fitting
                                    set to 'auto' to determine optimal setting
    backend        : None :       : backend to use for processing (see backend
                                    module for details). If backend==None, the
                                    backend set in config will be used.
    
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
    
    if delta == 'auto':
        delta = var.optimize_delta_bisection( Xpca[:,:,:], P, xvschema=xvschema.multitrial )
    
    # fit MVAR model
    A = var.fit( Xpca, P, delta )
    
    # residuals
    r = Xpca - var.predict( Xpca, A )

    # run on residuals ICA to estimate volume conduction    
    Mx, Ux = backend['ica'](cat_trials(r))
    
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
    
    
def cspvarica(X, cl, P, reducedim=np.inf, delta=0, backend=None):
    '''
    cspvarica( X, cl, P )
    cspvarica( X, cl, P, reducedim, delta, backend )
    
    Apply CSPVARICA to the data X. CSPVARICA performs the following steps:
        1. CSP transform of the data (with optional dimensionality reduction)
        2. Fitting a VAR model tho the data
        3. Decomposing the VAR model residuals with ICA
        4. Correcting the VAR coefficients
    
    Parameters     Default  Shape   Description
    --------------------------------------------------------------------------
    X              :      : N,M,T : 3d data matrix (N samples, M signals, T trials)
                          : N,M   : 2d data matrix (N samples, M signals)
    P              :      :       : VAR model order
    reducedim      :      : 0.99  : An integer number of 1 or greater is
                                    interpreted as the number of components to
                                    keep after applying the CSP.
    delta          : 0    :       : regularization parameter for VAR fitting
                                    set to 'auto' to determine optimal setting
    backend        : None :       : backend to use for processing (see backend
                                    module for details). If backend==None, the
                                    backend set in config will be used.
    
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
    
    if backend == None:
        backend = config.backend
    
    # pre-transform the data with CSP
    C, D, Xcsp = backend['csp'](X, cl, reducedim)
    M = C.shape[1]
    
    if delta == 'auto':
        delta = var.optimize_delta_bisection( Xcsp[:,:,:], P, xvschema=xvschema.multitrial )
    
    # fit MVAR model
    A = var.fit( Xcsp, P, delta )
    
    # residuals
    r = Xcsp - var.predict( Xcsp, A )

    # run on residuals ICA to estimate volume conduction    
    Mx, Ux = backend['ica'](cat_trials(r))
    
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
    
