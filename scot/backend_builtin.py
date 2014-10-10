# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

""" Use internally implemented functions as backend.
"""

import numpy as np
import scipy as sp

from . import config, datatools, pca, csp
from .var import VAR
from .external.infomax_ import infomax


def wrapper_infomax(data):
    """ Call binica for ICA calculation.
    """
    u = infomax(datatools.cat_trials(data)).T
    m = sp.linalg.pinv(u)
    return m, u

def wrapper_pca(x, reducedim):
    """ Call SCoT's PCA algorithm.
    """
    c, d = pca.pca(datatools.cat_trials(x), subtract_mean=False, reducedim=reducedim)
    y = datatools.dot_special(x, c)
    return c, d, y
    
def wrapper_csp(x, cl, reducedim):
    c, d = csp.csp(x, cl, numcomp=reducedim)
    y = datatools.dot_special(x,c)
    return c, d, y


backend = {
    'ica': wrapper_infomax,
    'pca': wrapper_pca,
    'csp': wrapper_csp,
    'var': VAR
}


def activate():
    config.backend = backend


activate()
