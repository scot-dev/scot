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