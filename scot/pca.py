# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013-2016 SCoT Development Team

"""Principal component analysis (PCA) implementation."""

import numpy as np
from .datatools import cat_trials


def pca_svd(x):
    """Calculate PCA using SVD.
    
    Parameters
    ----------
    x : ndarray, shape (channels, samples)
        Two-dimensional input data.
        
    Returns
    -------
    w : ndarray, shape (channels, channels)
        Eigenvectors (principal components) (in columns).
    s : ndarray, shape (channels,)
        Eigenvalues.
    """
    w, s, _ = np.linalg.svd(x, full_matrices=False)
    return w, s ** 2


def pca_eig(x):
    """Calculate PCA using eigenvalue decomposition.
    
    Parameters
    ----------
    x : ndarray, shape (channels, samples)
        Two-dimensional input data.

    Returns
    -------
    w : ndarray, shape (channels, channels)
        Eigenvectors (principal components) (in columns).
    s : ndarray, shape (channels,)
        Eigenvalues.
    """
    s, w = np.linalg.eigh(x.dot(x.T))
    return w, s


def pca(x, subtract_mean=False, normalize=False, sort_components=True,
        reducedim=None, algorithm=pca_eig):
    """Calculate principal component analysis (PCA).
    
    Parameters
    ----------
    x : ndarray, shape (trials, channels, samples) or (channels, samples)
        Input data.
    subtract_mean : bool, optional
        Subtract sample mean from x.
    normalize : bool, optional
        Normalize variances before applying PCA.
    sort_components : bool, optional
        Sort principal components in order of decreasing eigenvalues.
    reducedim : float or int or None, optional
        A value less than 1 is interpreted as the fraction of variance that
        should be retained in the data. All components that account for less
        than `1 - reducedim` of the variance are removed.
        An integer value of 1 or greater is interpreted as the number of
        (sorted) components to retain.
        If None, do not reduce dimensionality (i.e. keep all components).
    algorithm : func, optional
        Function to use for eigenvalue decomposition
        (:func:`pca_eig` or :func:`pca_svd`).
        
    Returns
    -------
    w : ndarray, shape (channels, components)
        PCA transformation matrix.
    v : ndarray, shape (components, channels)
        Inverse PCA transformation matrix.
    """

    x = np.asarray(x)
    if x.ndim == 3:
        x = cat_trials(x)

    if reducedim:
        sort_components = True

    if subtract_mean:
        x = x - np.mean(x, axis=1, keepdims=True)

    k, l = None, None
    if normalize:
        l = np.std(x, axis=1, ddof=1)
        k = np.diag(1.0 / l)
        l = np.diag(l)
        x = np.dot(k, x)

    w, latent = algorithm(x)

    # PCA is just a rotation, so inverse is equal to transpose
    v = w.T

    if normalize:
        w = np.dot(k, w)
        v = np.dot(v, l)

    latent /= sum(latent)

    if sort_components:
        order = np.argsort(latent)[::-1]
        w = w[:, order]
        v = v[order, :]
        latent = latent[order]

    if reducedim is not None:
        if reducedim < 1:
            selected = np.nonzero(np.cumsum(latent) < reducedim)[0]
            try:
                selected = np.concatenate([selected, [selected[-1] + 1]])
            except IndexError:
                selected = [0]
            if selected[-1] >= w.shape[1]:
                selected = selected[0:-1]
            w = w[:, selected]
            v = v[selected, :]
        else:
            w = w[:, :reducedim]
            v = v[:reducedim, :]

    return w, v
