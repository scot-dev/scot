# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013-2015 SCoT Development Team

""" Use scikit-learn routines as backend.
"""

from __future__ import absolute_import

import scipy as sp
from . import backend_builtin as builtin
from . import config, datatools
from .varbase import VARBase
from .datatools import atleast_3d


def wrapper_fastica(data):
    """ Call FastICA implementation from scikit-learn.
    """
    from sklearn.decomposition import FastICA
    ica = FastICA()
    ica.fit(datatools.cat_trials(data).T)
    u = ica.components_.T
    m = ica.mixing_.T
    return m, u


def wrapper_pca(x, reducedim):
    """ Call PCA implementation from scikit-learn.
    """
    from sklearn.decomposition import PCA
    pca = PCA(n_components=reducedim)
    pca.fit(datatools.cat_trials(x).T)
    d = pca.components_
    c = pca.components_.T
    y = datatools.dot_special(c.T, x)
    return c, d, y


class VAR(VARBase):
    """ Scikit-learn based implementation of VARBase.

    This class fits VAR models using various implementations of generalized linear model fitting available in scikit-learn.
    
    Parameters    
    ----------
    model_order : int
        Autoregressive model order
    fitobj : class, optional
        Instance of a linear model implementation.
    """
    def __init__(self, model_order, fitobj=None):
        VARBase.__init__(self, model_order)
        if fitobj is None:
            from sklearn.linear_model import LinearRegression
            fitobj = LinearRegression(fit_intercept=False)
        self.fitting_model = fitobj

    def fit(self, data):
        """ Fit VAR model to data.
        
        Parameters
        ----------
        data : array, shape (n_trials, n_channels, n_samples) or (n_channels, n_samples)
            Continuous or segmented data set.
            
        Returns
        -------
        self : :class:`VAR`
            The :class:`VAR` object.
        """
        data = atleast_3d(data)
        (x, y) = self._construct_eqns(data)
        self.fitting_model.fit(x, y)

        self.coef = self.fitting_model.coef_

        self.residuals = data - self.predict(data)
        self.rescov = sp.cov(datatools.cat_trials(self.residuals[:, :, self.p:]))

        return self


backend = builtin.backend.copy()
backend.update({
    'ica': wrapper_fastica,
    'pca': wrapper_pca,
    'var': VAR
})


def activate():
    """ Set backend attribute in the config module.
    """
    config.backend = backend


activate()
