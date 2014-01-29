# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

import numpy as np

from . import config
from .datatools import cat_trials, dot_special
from . import xvschema


def mvarica(x, var, cl=None, reducedim=0.99, optimize_var=False, backend=None, varfit='ensemble'):
    """ Performs joint VAR model fitting and ICA source separation.
    
    This function implements the MVARICA procedure [1]_.
    
    Parameters
    ----------
    x : array-like, shape = [n_samples, n_channels, n_trials] or [n_samples, n_channels]
        data set
    cl : list of valid dict keys
        Class labels associated with each trial.
    reducedim : {int, float, 'no_pca'}, optional
        A number of less than 1 in interpreted as the fraction of variance that should remain in the data. All
        components that describe in total less than `1-reducedim` of the variance are removed by the PCA step.
        An integer numer of 1 or greater is interpreted as the number of components to keep after applying the PCA.
        If set to 'no_pca' the PCA step is skipped.
    optimize_var : bool, optional
        Whether to call automatic optimization of the VAR fitting routine.
    backend : dict-like, optional
        Specify backend to use. When set to None the backend configured in config.backend is used.
    varfit : string
        Determines how to calculate the residuals for source decomposition.
        'ensemble' (default) fits one model to the whole data set,
        'class' fits a different model for each class, and
        'trial' fits a different model for each individual trial.
        
    Returns
    -------
    result : class
        A class with the following attributes is returned:
            
        +---------------+----------------------------------------------------------+
        | mixing        | Source mixing matrix                                     |
        +---------------+----------------------------------------------------------+
        | unmixing      | Source unmixing matrix                                   |
        +---------------+----------------------------------------------------------+
        | residuals     | Residuals of the VAR model(s) in source space            |
        +---------------+----------------------------------------------------------+
        | var_residuals | Residuals of the VAR model(s) in EEG space (before ICA)  |
        +---------------+----------------------------------------------------------+
        | c             | Noise covariance of the VAR model(s) in source space     |
        +---------------+----------------------------------------------------------+
        | b             | VAR model coefficients (source space)                    |
        +---------------+----------------------------------------------------------+
        | a             | VAR model coefficients (EEG space)                       |
        +---------------+----------------------------------------------------------+
        
    Notes
    -----
    MVARICA is performed with the following steps:        
    1. Optional dimensionality reduction with PCA
    2. Fitting a VAR model tho the data
    3. Decomposing the VAR model residuals with ICA
    4. Correcting the VAR coefficients
        
    References
    ----------
    .. [1] G. Gomez-Herrero et al. "Measuring directional coupling between EEG sources", NeuroImage, 2008
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

    if optimize_var:
        var.optimize(xpca)

    if varfit == 'trial':
        r = np.zeros(xpca.shape)
        for i in range(t):
            # fit MVAR model
            a = var.fit(xpca[:,:,i])
            # residuals
            r[:,:,i] = xpca[:,:,i] - var.predict(xpca[:,:,i])[:,:,0]
    elif varfit == 'class':
        r = np.zeros(xpca.shape)
        for i in np.unique(cl):
            mask = cl==i
            a = var.fit(xpca[:,:,mask])
            r[:,:,mask] = xpca[:,:,mask] - var.predict(xpca[:,:,mask])
    elif varfit == 'ensemble':
        # fit MVAR model
        a = var.fit(xpca)
        # residuals
        r = xpca - var.predict(xpca)
    else:
        raise InvalidArgument('unknown VAR fitting mode')

    # run on residuals ICA to estimate volume conduction    
    mx, ux = backend['ica'](cat_trials(r))

    # driving process
    e = dot_special(r, ux)

    # correct AR coefficients
    b = a.copy()
    for k in range(0, a.p):
        b.coef[:, k::a.p] = mx.dot(a.coef[:, k::a.p].transpose()).dot(ux).transpose()

    # correct (un)mixing matrix estimatees
    mx = mx.dot(d)
    ux = c.dot(ux)

    class Result:
        unmixing = ux
        mixing = mx
        residuals = e
        var_residuals = r
        c = np.cov(cat_trials(e), rowvar=False)

    Result.b = b
    Result.a = a
    Result.xpca = xpca
        
    return Result
    
    
def cspvarica(x, cl, var, reducedim=np.inf, optimize_var=False, backend=None, varfit='ensemble'):
    """
    cspvarica( x, cl, var )
    cspvarica( x, cl, var, reducedim, optimize_var, backend )
    
    Apply CSPVARICA to the data X. CSPVARICA performs the following steps:
        1. CSP transform of the data (with optional dimensionality reduction)
        2. Fitting a VAR model tho the data
        3. Decomposing the VAR model residuals with ICA
        4. Correcting the VAR coefficients
    
    Parameters     Default  Shape   Description
    --------------------------------------------------------------------------
    x              :      : n,m,t : 3d data matrix (n samples, m signals, t trials)
                          : n,m   : 2d data matrix (n samples, m signals)
    var            :      :       : Instance of class that represents VAR models.
    reducedim      :      : 0.99  : An integer number of 1 or greater is
                                    interpreted as the number of components to
                                    keep after applying the CSP.
    backend        : None :       : backend to use for processing (see backend
                                    module for details). If backend==None, the
                                    backend set in config will be used.
    varfit         :'ensemble':   : 'ensemble' (default) fits one VAR model to
                                    the whole data set.
                                    'class' fits one VAR model for each class.
                                    'trial' fits one VAR model for each trial.
    
    Output
    --------------------------------------------------------------------------
    b   Model coefficients: [B_0, B_1, ... B_P], each sub matrix B_k is of size M*M
    u   Unmixing matrix
    m   Mixing matrix
    e   Residual process
    c   Residual covariance matrix
    delta   Regularization parameter
    
    Note on the arrangement of model coefficients:
        B is of shape M, M*P, with sub matrices arranged as follows:
            b_00 b_01 ... b_0M
            b_10 b_11 ... b_1M
            .... ....     ....
            b_M0 b_M1 ... b_MM
        Each sub matrix b_ij is a column vector of length P that contains the
        filter coefficients from channel j (source) to channel i (sink).
    """
    
    x = np.atleast_3d(x)
    l, m, t = np.shape(x)
    
    if backend == None:
        backend = config.backend
    
    # pre-transform the data with CSP
    c, d, xcsp = backend['csp'](x, cl, reducedim)
    m = c.shape[1]
    
    if optimize_var:
        var.optimize(xcsp)

    if varfit == 'trial':
        r = np.zeros(xcsp.shape)
        for i in range(t):
            # fit MVAR model
            a = var.fit(xcsp[:,:,i])
            # residuals
            r[:,:,i] = xcsp[:,:,i] - var.predict(xcsp[:,:,i])[:,:,0]
    elif varfit == 'class':
        r = np.zeros(xcsp.shape)
        for i in np.unique(cl):
            mask = cl==i
            a = var.fit(xcsp[:,:,mask])
            r[:,:,mask] = xcsp[:,:,mask] - var.predict(xcsp[:,:,mask])
    elif varfit == 'ensemble':
        # fit MVAR model
        a = var.fit(xcsp)
        # residuals
        r = xcsp - var.predict(xcsp)
    else:
        raise InvalidArgument('unknown VAR fitting mode')

    # run on residuals ICA to estimate volume conduction    
    mx, ux = backend['ica'](cat_trials(r))

    # driving process
    e = dot_special(r, ux)

    # correct AR coefficients
    b = a.copy()
    for k in range(0,a.p):
        b.coef[:, k::a.p] = mx.dot(a.coef[:, k::a.p].transpose()).dot(ux).transpose()
    
    # correct (un)mixing matrix estimatees
    mx = mx.dot(d)
    ux = c.dot(ux)
    
    class Result:
        unmixing = ux
        mixing = mx
        residuals = e
        var_residuals = r
        c = np.cov(cat_trials(e), rowvar=False)
    Result.b = b
    Result.a = a
    Result.xcsp = xcsp

    return Result

    
