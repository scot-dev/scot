# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

from .. import config

from .. import datatools

from sklearn.decomposition import FastICA, PCA

def wrapper_fastica(data):
    ica = FastICA()
    ica.fit(datatools.cat_trials(data))
    U = ica.components_.T
    M = ica.mixing_.T
    return M, U

def wrapper_pca(X, retain_variance, numcomp):
    if retain_variance != None and numcomp != None:
        raise AttributeError('Conflicting parameters: retain_variance and numcomp. At least one must be None.')
    n_components = numcomp
    if retain_variance:
        if retain_variance >= 1:
            retain_variance = None
        n_components = retain_variance
    pca = PCA(n_components=n_components)
    Y = pca.fit(datatools.cat_trials(X))
    D = pca.components_
    C = pca.components_.T
    Y = datatools.dot_special(X,C)
    return C, D, Y
    
backend = {
    'ica': wrapper_fastica,
    'pca': wrapper_pca
    }
    
def activate( ):
    config.backend = backend
    
activate()