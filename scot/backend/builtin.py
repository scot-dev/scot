# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

import numpy as np

from .. import config

from .. import datatools

from ..builtin import binica, pca, csp


def wrapper_binica(data):
    w, s = binica.binica(datatools.cat_trials(data))
    u = s.dot(w)
    m = np.linalg.inv(u)
    return m, u

def wrapper_pca(x, reducedim):
    c, d = pca.pca(datatools.cat_trials(x), subtract_mean=False, reducedim=reducedim)
    y = datatools.dot_special(x, c)
    return c, d, y
    
def wrapper_csp(x, cl, reducedim):
    c, d = csp.csp(x, cl, numcomp=reducedim)
    y = datatools.dot_special(x,c)
    return c, d, y


backend = {
    'ica': wrapper_binica,
    'pca': wrapper_pca,
    'csp': wrapper_csp
}


def activate():
    config.backend = backend


activate()