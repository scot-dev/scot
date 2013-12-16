# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

"""principal component analysis (PCA) implementation"""

import numpy as np
from ..datatools import cat_trials


def pca_svd(data):
    """calculate PCA from SVD (observations in rows)"""

    (w, s, v) = np.linalg.svd(data.transpose())

    return w, s ** 2


def pca_eig(x):
    """calculate PCA as eigenvalues of the covariance (observations in rows)"""

    [w, v] = np.linalg.eigh(x.transpose().dot(x))

    return v, w


def pca(x, subtract_mean=False, normalize=False, sort_components=True, reducedim=None, algorithm=pca_eig):
    """
    pca( x, subtract_mean=False,
            normalize=False,
            sort_components=True,
            retain_variance=None,
            algorithm=pcaEIG ):

    calculate principal component analysis (PCA).

    Parameters     Default  Shape   Description
    --------------------------------------------------------------------------
    x              :      : n,m,T : 3d data matrix (n samples, m signals, T trials)
                          : n,m   : 2d data matrix (n samples, m signals)
    subtract_mean  : False:       : If True, the sample mean is subtracted from x
    normalize      : False:       : If True, the data is normalized to unit variance
    sort_components: True :       : If True, components are sorted by decreasing variance
    reducedim      : None :       : a number less than 1 is interpreted as the
                                    fraction of variance that should remain in
                                    the data. All components that describe in
                                    total less than 1-retain_variance of the
                                    variance in the data are removed by the PCA.
                                    An integer number of 1 or greater is
                                    interpreted as the number of components to
                                    keep after applying the PCA.
                                    None or a number greater than m does not
                                    remove components.
    numcomp        : None :       : Select numcomp components wtih highest variance
    algorithm      : pcaEIG :     : which function to call for eigenvector estimation

    Output
    --------------------------------------------------------------------------
    w   PCA weights      y = x * w
    v   inverse weights  x = y * v
    """

    x = cat_trials(np.atleast_3d(x))

    if reducedim:
        sort_components = True

    if subtract_mean:
        for i in range(np.shape(x)[1]):
            x[:, i] -= np.mean(x[:, i])

    k, l = None, None
    if normalize:
        l = np.std(x, 0, ddof=1)
        k = np.diag(1.0 / l)
        l = np.diag(l)
        x = x.dot(k)

    w, latent = algorithm(x)

    #v = np.linalg.inv(w)
    # PCA is just a rotation, so inverse is equal transpose...
    v = w.T

    if normalize:
        w = k.dot(w)
        v = v.dot(l)

    latent /= sum(latent)

    if sort_components:
        order = np.argsort(latent)[::-1]
        w = w[:, order]
        v = v[order, :]
        latent = latent[order]

    if reducedim and reducedim < 1:
        selected = np.nonzero(np.cumsum(latent) < reducedim)[0]
        try:
            selected = np.concatenate([selected, [selected[-1] + 1]])
        except IndexError:
            selected = [0]
        if selected[-1] >= w.shape[1]:
            selected = selected[0:-1]
        w = w[:, selected]
        v = v[selected, :]

    if reducedim and reducedim >= 1:
        w = w[:, np.arange(reducedim)]
        v = v[np.arange(reducedim), :]

    return w, v
