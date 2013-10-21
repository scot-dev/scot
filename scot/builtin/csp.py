# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

"""common spatial patterns (CSP) implementation"""

import numpy as np
from scipy.linalg import eig
    

def csp( X, cl, numcomp=np.inf ):    
    '''
    csp( X, cl, halfnumcomp ):
        
    Create common spatial patterns (CSP): spatial filters for maximizing
    inter-class variance. Only supports two classes!
    
    Parameters     Default  Shape   Description
    --------------------------------------------------------------------------
    X              :      : N,M,T : 3d data matrix (N samples, M signals, T trials)
    cl             :      : T     : list of class labels (one of the labels must be 1)
    
    Output
    --------------------------------------------------------------------------
    W   CSP filter                       Y = X * W
    V   inverse (reconstruction) filter  X = Y * V
    '''
    
    # TODO: optimization:
    #   1. subtracting the mean is not required
    #   2. normalizing each trial is not required, if instead we normalize by the diagonal of the covariance matrix
    #   3. then X.copy() is no longer required.
    
    # TODO: normalize by trace
    
    # make sure we don't mess with the outside X
    X = np.atleast_3d(X).copy()    
    cl = np.asarray(cl).ravel()
    
    N, M, T = X.shape
    
    if T != cl.size:
        raise AttributeError('CSP only works with multiple classes. Number of elemnts in cl (%d) must equal 3rd dimension of X (%d)'%(cl.size, T))
        
    # subtract mean for each trial separately
    for t in range(T):
        for m in range(M):
            X[:,m,t] -= np.mean(X[:,m,t])
            
    # normalize each trial separately
    for t in range(T):
        K = np.diag(1.0 / np.std(X[:,:,t], 0, ddof=1))
        X[:,:,t] = X[:,:,t].dot(K)
        
    X1 = X[:,:,cl==1]
    X2 = X[:,:,cl!=1]
    
    Sigma1 = np.zeros((M,M))
    for t in range(X1.shape[2]):
        Sigma1 += np.cov(X1[:,:,t].transpose()) / X1.shape[2]
    Sigma1 /= Sigma1.trace()
    
    Sigma2 = np.zeros((M,M))
    for t in range(X2.shape[2]):
        Sigma2 += np.cov(X2[:,:,t].transpose()) / X2.shape[2]
    Sigma2 /= Sigma2.trace()
        
    E, W = eig(Sigma1, Sigma1 + Sigma2, overwrite_a=True, overwrite_b=True, check_finite=False)
        
    order = np.argsort(E)[::-1]
    W = W[:,order]
    E = E[:,order]
        
    V = np.linalg.inv(W)
   
    # subsequently remove unwanted components from the middle of W and V 
    while W.shape[1] > numcomp:
        i = int(np.floor(W.shape[1]/2))
        W = np.delete(W,i,1)
        V = np.delete(V,i,0)
        
    return W, V
