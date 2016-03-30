# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013-2016 SCoT Development Team

"""Use mne-python routines as backend."""

from __future__ import absolute_import
import scipy as sp

from . import datatools
from . import backend
from . import backend_builtin as builtin


def generate():
    from mne.preprocessing.infomax_ import infomax

    def wrapper_infomax(data, random_state=None):
        """Call Infomax for ICA calculation."""
        u = infomax(datatools.cat_trials(data).T, extended=True,
                    random_state=random_state).T
        m = sp.linalg.pinv(u)
        return m, u

    def wrapper_csp(x, cl, reducedim):
        """Call MNE CSP algorithm."""
        from mne.decoding import CSP
        csp = CSP(n_components=reducedim, cov_est="epoch")
        y = csp.fit_transform(x, cl)
        return csp.filters_, csp.patterns_, y

    backend = builtin.generate()
    backend.update({'ica': wrapper_infomax, 'csp': wrapper_csp})
    return backend


backend.register('mne', generate)
