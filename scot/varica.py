# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

import numpy as np

from . import config
from .datatools import cat_trials, dot_special
from . import xvschema
from . import var


def mvarica(x, p, reducedim=0.99, delta=0, backend=None):
    """
    mvarica( x, p )
    mvarica( x, p, retain_variance, delta )
    mvarica( x, p, numcomp, delta )

    Apply MVARICA to the data x. MVARICA performs the following steps:
        1. Optional dimensionality reduction with PCA
        2. Fitting a VAR model tho the data
        3. Decomposing the VAR model residuals with ICA
        4. Correcting the VAR coefficients

    Parameters     Default  Shape   Description
    --------------------------------------------------------------------------
    x              :      : n,m,t : 3d data matrix (n samples, m signals, t trials)
                          : n,m   : 2d data matrix (n samples, m signals)
    p              :      :       : VAR model order
    reducedim      : 0.99 :       : a number less than 1 is interpreted as the
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
    b   Model coefficients: [B_0, B_1, ... B_P], each sub matrix B_k is of size m*m
    U   Unmixing matrix
    m   Mixing matrix
    e   Residual process
    c   Residual covariance matrix
    delta   Regularization parameter

    Note on the arrangement of model coefficients:
        b is of shape m, m*p, with sub matrices arranged as follows:
            b_00 b_01 ... b_0m
            b_10 b_11 ... b_1m
            .... ....     ....
            b_m0 b_m1 ... b_mm
        Each sub matrix b_ij is a column vector of length p that contains the
        filter coefficients from channel j (source) to channel i (sink).
    """

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

    if delta == 'auto':
        delta = var.optimize_delta_bisection(xpca[:, :, :], p, xvschema=xvschema.multitrial)

    #r = np.zeros(xpca.shape)
    #for i in range(t):
    #    # fit MVAR model
    #    a = var.fit(xpca[:,:,i], p, delta)
    #
    #    # residuals
    #    r[:,:,i] = xpca[:,:,i] - var.predict(xpca[:,:,i], a)[:,:,0]

    # fit MVAR model
    a = var.fit(xpca, p, delta)

    # residuals
    r = xpca - var.predict(xpca, a)

    # run on residuals ICA to estimate volume conduction    
    mx, ux = backend['ica'](cat_trials(r))

    # driving process
    e = dot_special(r, ux)

    # correct AR coefficients
    b = np.zeros(a.shape)
    for k in range(0, p):
        b[:, k::p] = mx.dot(a[:, k::p].transpose()).dot(ux).transpose()

    # correct (un)mixing matrix estimatees
    mx = mx.dot(d)
    ux = c.dot(ux)

    class Result:
        unmixing = ux
        mixing = mx
        residuals = e
        var_residuals = r
        c = np.cov(cat_trials(e), rowvar=False)

    Result.delta = delta
    Result.b = b
    Result.a = a
    Result.xpca = xpca

    return Result
