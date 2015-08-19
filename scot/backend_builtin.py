# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013-2015 SCoT Development Team

""" Use internally implemented functions as backend.
"""

from __future__ import absolute_import

from . import backend, config


def generate():
    import scipy as sp

    from . import datatools, pca, csp
    from .var import VAR
    from .external.infomax_ import infomax

    def wrapper_infomax(data):
        """ Call Infomax (from MNE) for ICA calculation.
        """
        u = infomax(datatools.cat_trials(data).T).T
        m = sp.linalg.pinv(u)
        return m, u

    def wrapper_pca(x, reducedim):
        """ Call SCoT's PCA algorithm.
        """
        c, d = pca.pca(datatools.cat_trials(x),
                       subtract_mean=False, reducedim=reducedim)
        y = datatools.dot_special(c.T, x)
        return c, d, y

    def wrapper_csp(x, cl, reducedim):
        c, d = csp.csp(x, cl, numcomp=reducedim)
        y = datatools.dot_special(c.T, x)
        return c, d, y

    return {
        'ica': wrapper_infomax,
        'pca': wrapper_pca,
        'csp': wrapper_csp,
        'var': VAR
    }


backend.register('builtin', generate)
