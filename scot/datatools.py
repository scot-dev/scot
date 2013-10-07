# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

"""tools for basic data manipulation"""

import numpy as np

def cut_segments(rawdata, tr, start, stop):  
    '''
    X = cut_segments(rawdata, tr, start, stop):
        
    Cut continuous signal into segments of length L = stop - start.
    
    Parameters     Default  Shape   Description
    --------------------------------------------------------------------------
    rawdata        :      : N,M   : continuous signal data with N samples and M signals
    tr             :      : T     : list of trigger positions
    start          :      : 1     : window start (offset relative to trigger)
    stop           :      : 1     : window end (offset relative to trigger)
    
    Output
    --------------------------------------------------------------------------
    X              :      : L,M,T : 3d data matrix (L = stop - start)
    '''
    rawdata = np.atleast_2d(rawdata)
    tr = np.array(tr, dtype='int').ravel()
    win = range(start, stop)
    return np.dstack([rawdata[tr[t]+win,:] for t in range(len(tr))])
    
def cat_trials(X):
    '''
    Y = cat_trials(X):
        
    Concatenate trials along time axis.
    
    Parameters     Default  Shape   Description
    --------------------------------------------------------------------------
    X              :      : L,M,T : 3d data matrix (L samples, M signals, T trials)
    
    Output
    --------------------------------------------------------------------------
    Y              :      : L*T,M : 2d data matrix with trials concatenated along time axis
    '''
    X = np.atleast_3d(X)
    T = X.shape[2]
    return np.squeeze(np.vstack(np.dsplit(X,T)), axis=2)

def dot_special(X, A):
    '''
    Y = dot_special(X, A):        
        
    Dot product of 3D matrix X with 2D matrix A.

    This is equivalent to writing
        Y = np.dstack([X[:,:,i].dot(A) for i in range(X.shape[2])])
    
    Computes Y[:,:,i] = X[:,:,i].dot(A) for every i.
    
    Parameters     Default  Shape   Description
    --------------------------------------------------------------------------
    X              :      : L,M,T : 3d data matrix (L samples, M signals, T trials)
    A              :      : M,N   : 2d matrix to transform X with
    
    Output
    --------------------------------------------------------------------------
    Y              :      : L,N,T : 3d data matrix Y[:,:,i] = X[:,:,i].dot(A)
    '''
    X = np.atleast_3d(X)
    A = np.atleast_2d(A)
    return np.dstack([X[:,:,i].dot(A) for i in range(X.shape[2])])
    