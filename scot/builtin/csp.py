# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

"""common spatial patterns (CSP) implementation"""

import numpy as np
from scipy.linalg import eig
from itertools import combinations


def csp(x, cl, numcomp=None, mode='ova'):
    """ Calculate common spatial patterns (CSP)

    Parameters
    ----------
    x : array-like, shape = [n_samples, n_channels, n_trials] or [n_samples, n_channels]
        EEG data set
    cl : list of valid dict keys
        Class labels associated with each trial.
    numcomp : {int}, optional
        Number of patterns to keep after applying the CSP. If `numcomp` is greater than n_channels, all n_channels
        patterns are returned.
    mode : str, optional
        Determines what multi-class heuristic to use. Can be 'ova' or 'ovr' for one-versus-rest, or 'pairwise'.

    Returns
    -------
    w : array, shape = [n_channels, n_components]
        CSP weight matrix
    v : array, shape = [n_components, n_channels]
        CSP projection matrix
    """

    x = np.atleast_3d(x)
    cl = np.asarray(cl).ravel()

    n, m, t = x.shape

    if t != cl.size:
        raise AttributeError('CSP only works with multiple classes. Number of'
                             ' elemnts in cl (%d) must equal 3rd dimension of X (%d)' % (cl.size, t))

    labels = np.unique(cl)
    n_classes = labels.size

    if numcomp is None:
        if mode in ('ova', 'ovr'):
            numcomp = m
        elif mode == 'pairwise':
            numcomp = max(m, n_classes*(n_classes-1)/2)
    elif numcomp % n_classes != 0:
        print('warning: numcomp=={} is not a multiple of n_classes=={}'.format(numcomp, n_classes))

    # calculate covariance for each class
    sigma = {}
    for c in labels:
        xc = x[:, :, cl == c]
        sigma_c = np.zeros((m, m))
        for t in range(xc.shape[2]):
            sigma_c += np.cov(xc[:, :, t].transpose()) / xc.shape[2]
        sigma[c] = sigma_c / sigma_c.trace()

    w = {}
    if mode in ('ova', 'ovr'):
        # one-versus-rest
        sigma_pooled = np.zeros((m, m))
        for c in labels:
            sigma_pooled += sigma[c]
        for c in labels:
            evals, evecs = eig(sigma[c], sigma_pooled, overwrite_a=True, overwrite_b=False, check_finite=False)
            w[c] = evecs[:, np.argsort(evals)[::-1]]
    elif mode == 'pairwise':
        # one-versus-one
        for c, d in combinations(labels, 2):
            evals, evecs = eig(sigma[c], sigma[c] + sigma[d], overwrite_a=False, overwrite_b=True, check_finite=False)
            w[(c, d)] = evecs[:, np.argsort(evals)[::-1]]

    v = {k_: np.linalg.inv(w_) for k_, w_ in w.items()}

    # select components
    ws = np.empty((m, numcomp))
    vs = np.empty((numcomp, m))
    component, classcomp = 0, 0
    while component < numcomp:
        for k in w.keys():
            ws[:, component] = w[k][:, classcomp]
            vs[component, :] = v[k][classcomp, :]
            component += 1
            if component >= numcomp:
                break
        classcomp += 1

    if False:
        #raise ValueError('unknown mode: {}'.format(mode))

        x1 = x[:, :, cl == labels[0]]
        x2 = x[:, :, cl == labels[1]]

        sigma1 = np.zeros((m, m))
        for t in range(x1.shape[2]):
            sigma1 += np.cov(x1[:, :, t].transpose()) / x1.shape[2]
        sigma1 /= sigma1.trace()

        sigma2 = np.zeros((m, m))
        for t in range(x2.shape[2]):
            sigma2 += np.cov(x2[:, :, t].transpose()) / x2.shape[2]
        sigma2 /= sigma2.trace()

        e, w = eig(sigma1, sigma1 + sigma2, overwrite_a=True, overwrite_b=True, check_finite=False)

        order = np.argsort(e)[::-1]
        w = w[:, order]
        e = e[order]

        v = np.linalg.inv(w)

        # select components
        ws = np.empty((m, numcomp))
        vs = np.empty((numcomp, m))
        component, classcomp = 0, 0
        while component < numcomp:
            for cmp in [classcomp, -classcomp-1]:    # alternately take from the beginning and the end
                ws[:, component] = w[:, cmp]
                vs[component, :] = v[cmp, :]
                component += 1
                if component >= numcomp:
                    break
            classcomp += 1

        # subsequently remove unwanted components from the middle of w and v
        #while w.shape[1] > numcomp:
        #    i = int(np.floor(w.shape[1]/2))
        #    w = np.delete(w, i, 1)
        #    v = np.delete(v, i, 0)

    return ws, vs


if __name__ == '__main__':

    def generate_covsig(covmat, n):
        """generate pseudorandom stochastic signals with covariance matrix covmat"""

        covmat = np.atleast_2d(covmat)
        m = covmat.shape[0]
        l = np.linalg.cholesky(covmat)

        x = np.random.randn(m, n)

        # matrix to make cov(x) = I
        d = np.linalg.inv(np.linalg.cholesky(np.atleast_2d(np.cov(x))))

        x = l.dot(d).dot(x)

        return x.T

    from scot.datatools import dot_special, cat_trials
    from math import cos, sin

    def rot(a, angles):
        t = angles[0]
        a = np.dot(a, [[cos(t), -sin(t), 0], [sin(t), cos(t), 0], [0, 0, 1]])
        t = angles[1]
        a = np.dot(a, [[cos(t), 0, -sin(t)], [0, 1, 0], [sin(t), 0, cos(t)]])
        t = angles[2]
        a = np.dot(a, [[1, 0, 0], [0, cos(t), -sin(t)], [0, sin(t), cos(t)]])
        return a

    A = rot(generate_covsig([[10, 0, 0], [0, 10, 0], [0, 0, 1]], 500), np.random.rand(3))
    B = rot(generate_covsig([[10, 0, 0], [0, 10, 0], [0, 0, 1]], 500), np.random.rand(3))
    C = rot(generate_covsig([[10, 0, 0], [0, 10, 0], [0, 0, 1]], 500), np.random.rand(3))

    #A = generate_covsig([[10,0,0],[0,1,0],[0,0,1]], 500)
    #B = generate_covsig([[1,0,0],[0,10,0],[0,0,1]], 500)
    #C = generate_covsig([[1,0,0],[0,1,0],[0,0,10]], 500)

    X = np.dstack([A,B,C])
    W, V = csp(X, [1,2,3], mode='pairwise')
    print(W)
    W, V = csp(X, [1,2,3], mode='ova')
    print(W)

    print(W.dot(np.var(cat_trials(X), axis=0)))
    print(np.var(cat_trials(X), axis=0).dot(W))

    Y = dot_special(X, W)
    Y = cat_trials(Y)
    Y **= 2

    import scipy.signal as sps

    Y = sps.filtfilt(np.ones(100), [1], Y.T).T

    import matplotlib.pyplot as plt
    plt.plot(Y)
    plt.show()