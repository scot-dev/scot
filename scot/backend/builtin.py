# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

import numpy as np

from .. import config

from .. import datatools

from ..builtin import binica, pca, csp

def wrapper_binica(data):
    W, S = binica.binica(datatools.cat_trials(data))
    U = S.dot(W)
    M = np.linalg.inv(U)
    return M, U  

def wrapper_pca(X, reducedim):
    C, D = pca.pca(datatools.cat_trials(X), subtract_mean=False, reducedim=reducedim)
    Y = datatools.dot_special(X, C)
    return C, D, Y
    
def wrapper_csp(X, cl, reducedim):
    C, D = csp.csp( X, cl, numcomp=reducedim )
    Y = datatools.dot_special(X,C)
    return C, D, Y

    
backend = {
    'ica': wrapper_binica,
    'pca': wrapper_pca,
    'csp': wrapper_csp
    }
    
def activate( ):
    config.backend = backend
    
activate()