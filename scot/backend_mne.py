# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013-2016 SCoT Development Team

"""Use mne-python routines as backend."""

from __future__ import absolute_import
import scipy as sp

from . import datatools
from . import backend
from . import backend_builtin as builtin

try:
    import mne
except ImportError:
    mne = None


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
        csp = CSP(n_components=reducedim, cov_est="epoch", reg="ledoit_wolf")
        csp.fit(x, cl)
        c, d = csp.filters_.T[:, :reducedim], csp.patterns_[:reducedim, :]
        y = datatools.dot_special(c.T, x)
        return c, d, y

    backend = builtin.generate()
    backend.update({'ica': wrapper_infomax, 'csp': wrapper_csp})
    return backend


if mne is not None:
    backend.register('mne', generate)
