# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

""" Utility functions """

import numpy as np


def cuthill_mckee(matrix):
    """ Cuthill-McKee algorithm
        Permute a symmetric binary matrix into a band matrix form with a small bandwidth.
    """
    matrix = np.atleast_2d(matrix)
    n, m = matrix.shape
    assert(n == m)

    # make sure the matrix is really symmetric. This is equivalent to
    # converting a directed adjacency matrix into a undirected adjacency matrix.
    matrix = np.logical_or(matrix, matrix.T)

    degree = np.sum(matrix, 0)
    order = [np.argmin(degree)]

    for i in range(n):
        adj = np.nonzero(matrix[order[i]])[0]
        adj = [a for a in adj if a not in order]
        if not adj:
            idx = [i for i in range(n) if i not in order]
            order.append(idx[np.argmin(degree[idx])])
        else:
            if len(adj) == 1:
                order.append(adj[0])
            else:
                adj = np.asarray(adj)
                i = adj[np.argsort(degree[adj])]
                order.extend(i.tolist())
        if len(order) == n:
            break

    return order


def acm(x, l):
    """ calculate the autocovariance matrix at lag l
    """
    x = np.atleast_3d(x)

    if l > x.shape[0]-1:
        raise AttributeError("lag exceeds data length")

    ## subtract mean from each trial
    #for t in range(x.shape[2]):
    #    x[:, :, t] -= np.mean(x[:, :, t], axis=0)

    if l == 0:
        a, b = x, x
    else:
        a = x[l:, :, :]
        b = x[0:-l, :, :]

    c = np.zeros((x.shape[1], x.shape[1]))
    for t in range(x.shape[2]):
        c += a[:, :, t].T.dot(b[:, :, t]) / x.shape[0]
    c /= x.shape[2]

    return c