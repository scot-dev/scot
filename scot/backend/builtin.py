# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

import numpy as np

from .. import config

from .. import datatools

from ..builtin import binica, pca

def wrapper_binica(data):
    W, S = binica.binica(datatools.cat_trials(data))
    U = S.dot(W)
    M = np.linalg.inv(U)
    return M, U  

def wrapper_pca(X, retain_variance, numcomp):
    C, D = pca.pca( datatools.cat_trials(X), subtract_mean=False, retain_variance=retain_variance, numcomp=numcomp )
    Y = datatools.dot_special(X,C)
    return C, D, Y
    
def activate( ):
    config.backend['ica'] = wrapper_binica
    config.backend['pca'] = wrapper_pca
    
activate()
