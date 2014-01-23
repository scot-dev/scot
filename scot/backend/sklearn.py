# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

from sklearn.decomposition import FastICA, PCA
from sklearn import linear_model
import scipy as sp

from . import builtin
from . import sklearn_utils

from .. import config
from .. import datatools
from ..var import VARBase


def wrapper_fastica(data):
    ica = FastICA()
    ica.fit(datatools.cat_trials(data))
    u = ica.components_.T
    m = ica.mixing_.T
    return m, u


def wrapper_pca(x, reducedim):
    pca = PCA(n_components=reducedim)
    pca.fit(datatools.cat_trials(x))
    d = pca.components_
    c = pca.components_.T
    y = datatools.dot_special(x, c)
    return c, d, y


class VAR(VARBase):
    def __init__(self, model_order, fitobj=linear_model.LinearRegression()):
        """ Create a new VAR model instance.

            Parameters     Default  Shape   Description
            --------------------------------------------------------------------------
            model_order    :      :       : Autoregressive model order
            fitobj         :      :       : Instance of a linear regression model.
                                            Default: sklearn.linear_model.LinearRegression()
        """
        VARBase.__init__(self, model_order)
        self.fitting_model = fitobj

    def fit(self, data):
        data = sp.atleast_3d(data)
        (x, y) = self._construct_eqns(data)
        self.fitting_model.fit(x, y)

        self.coef = self.fitting_model.coef_

        self.residuals = data - self.predict(data)
        self.rescov = sp.cov(datatools.cat_trials(self.residuals), rowvar=False)

        return self


backend = builtin.backend.copy()
backend.update({
    'ica': wrapper_fastica,
    'pca': wrapper_pca,
    'var': VAR,
    'utils': sklearn_utils
})


def activate():
    config.backend = backend


activate()