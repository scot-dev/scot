# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

from __future__ import absolute_import

from sklearn.decomposition import FastICA, PCA

from .. import config
from .. import datatools


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


backend = {
    'ica': wrapper_fastica,
    'pca': wrapper_pca
}


def activate():
    config.backend = backend


activate()