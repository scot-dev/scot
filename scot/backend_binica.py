# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

""" Use internally implemented functions as backend.
"""

import numpy as np

from . import backend_builtin as builtin
from . import config, datatools, binica


def wrapper_binica(data):
    """ Call binica for ICA calculation.
    """
    w, s = binica.binica(datatools.cat_trials(data))
    u = s.dot(w)
    m = np.linalg.inv(u)
    return m, u


backend = builtin.backend.copy()
backend.update({'ica': wrapper_binica})


def activate():
    config.backend = backend


activate()
