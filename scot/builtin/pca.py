# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

"""principal component analysis (PCA) implementation"""

import numpy as np
from ..datatools import cat_trials

def pcaSVD( X ):
    '''calculate PCA from SVD (observations in rows)'''    
    
    (W,S,V) = np.linalg.svd( X.transpose() )
    
    return W, S**2
    

def pcaEIG( X ):
    '''calculate PCA as eigenvalues of the covariance (observations in rows)'''
            
    [W,V] = np.linalg.eigh( X.transpose().dot(X) )
    
    return V, W
    

def pca( X, subtract_mean=False, normalize=False, sort_components=True, reducedim=None, algorithm=pcaEIG ):    
    '''
    pca( X, subtract_mean=False, 
            normalize=False, 
            sort_components=True, 
            retain_variance=None, 
            algorithm=pcaEIG ):
        
    calculate principal component analysis (PCA).
    
    Parameters     Default  Shape   Description
    --------------------------------------------------------------------------
    X              :      : N,M,T : 3d data matrix (N samples, M signals, T trials)
                          : N,M   : 2d data matrix (N samples, M signals)
    subtract_mean  : False:       : If True, the sample mean is subtracted from X
    normalize      : False:       : If True, the data is normalized to unit variance
    sort_components: True :       : If True, components are sorted by decreasing variance
    reducedim      : None :       : A number less than 1 is interpreted as the
                                    fraction of variance that should remain in
                                    the data. All components that describe in
                                    total less than 1-retain_variance of the
                                    variance in the data are removed by the PCA.
                                    An integer number of 1 or greater is
                                    interpreted as the number of components to
                                    keep after applying the PCA.
                                    None or a number greater than M does not
                                    remove components.
    numcomp        : None :       : Select numcomp components wtih highest variance
    algorithm      : pcaEIG :     : which function to call for eigenvector estimation
    
    Output
    --------------------------------------------------------------------------
    W   PCA weights      Y = X * W
    V   inverse weights  X = Y * V
    '''
    
    X = cat_trials(np.atleast_3d(X))
    
    if reducedim:
        sort_components = True
    
    if subtract_mean:
        for i in range(np.shape(X)[1]):
            X[:,i] -= np.mean(X[:,i])
            
    if normalize:
        L = np.std(X, 0, ddof=1)
        K = np.diag(1.0 / L)
        L = np.diag(L)
        X = X.dot(K)
        
    W, latent = algorithm( X )
        
    #V = np.linalg.inv(W)
    # PCA is just a rotation, so inverse is equal transpose...
    V = W.T
    
    if normalize:
        W = K.dot(W)
        V = V.dot(L)
        
    latent /= sum(latent)
        
    if sort_components:        
        order = np.argsort(latent)[::-1]
        W = W[:,order]
        V = V[order,:]
        latent = latent[:,order]
    
    if reducedim and reducedim < 1:
        selected = np.nonzero(np.cumsum(latent)<reducedim)[0]
        try:
            selected = np.concatenate( [selected, [selected[-1]+1]] )
        except IndexError:
            selected = [0]
        if selected[-1] >= W.shape[1]:
            selected = selected[0:-1]        
        W = W[:,selected]
        V = V[selected,:]
        
        
    if reducedim and reducedim >= 1:
        W = W[:,np.arange(reducedim)]
        V = V[np.arange(reducedim),:]
        
    return W, V
