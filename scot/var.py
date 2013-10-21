# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

""" vector autoregressive (VAR) model implementation"""

import numbers
import numpy as np
import scipy as sp
import scipy.linalg
from functools import partial
from . import datatools
from . import xvschema

class defaults:
    xvschema = xvschema.multitrial

def fit( data, P, delta=None, return_residuals=False, return_covariance=False ):
    '''
    fit( data, P )
    fit( data, P, delta )
        
    Fit least squares estimate of vector autoregressive (VAR) model with
    order P to the data.
    
    If sqrtdelta is provited and nonzero, the least squares estimation is
    regularized with ridge regression.
    
    Parameters       Default  Shape   Description
    --------------------------------------------------------------------------
    data             :      : N,M,T : 3d data matrix (N samples, M signals, T trials)
                            : N,M   : 2d data matrix (N samples, M signals)
    P                :      :       : Model order
    delta            : None :       : regularization parameter
    return_residuals : False :      : if True, also return model residuals
    return_covariance: False :      : if True, also return covariance
    
    Output
    --------------------------------------------------------------------------
    B   Model coefficients: [B_0, B_1, ... B_P], each sub matrix B_k is of size M*M
    res (optional) Model residuals: (same shape as data), note that
        the first P residuals are invalid.
    C   (optional) Covariance of residuals
    
    Note on the arrangement of model coefficients:
        B is of shape M, M*P, with sub matrices arranged as follows:
            b_00 b_01 ... b_0M
            b_10 b_11 ... b_1M
            .... ....     ....
            b_M0 b_M1 ... b_MM
        Each sub matrix b_ij is a column vector of length P that contains the
        filter coefficients from channel j (source) to channel i (sink).
    '''
    data = np.atleast_3d(data)
    (L,M,T) = np.shape( data )
    
    if delta == 0 or delta == None:
        # normal least squares
        (X,y) = __construct_eqns( data, P )
    else:    
        # regularized least squares
        (X,y) = __construct_eqns_RLS( data, P, delta )
    
    (B, res, rank, s) = np.linalg.lstsq( X, y )
    B = B.transpose()
    
    if return_residuals or return_covariance:
        result = [B]
        res = data - predict( data, B )
    else:
        return B
        
    if return_residuals:
        result.append(res)
        
    if return_covariance:
        C = np.cov(datatools.cat_trials(res), rowvar=False)
        result.append(C)
        
    return tuple(result)
   
   
    
def fit_multiclass( data, cl, P, delta=None, return_residuals=False, return_covariance=False ):
    '''
    fit_multiclass( data, cl, P )
    fit_multiclass( data, cl, P, delta )
    
    Fits a separate autoregressive model for each class.
    
    If sqrtdelta is provited and nonzero, the least squares estimation is
    regularized with ridge regression.
    
    Parameters     Default  Shape   Description
    --------------------------------------------------------------------------
    data             :      : N,M,T : 3d data matrix (N samples, M signals, T trials)
    cl               :      : T     : class label for each trial
    P                :      :       : Model order can be scalar (same model order for each class)
                                      or a dictionary that contains a key for each unique class label
                                      to specify the model order for each class individually.
    delta            : None :       : regularization parameter
    return_residuals : False :      : if True, also return model residuals
    return_covariance: False :      : if True, also return dictionary of covariances
    
    Output
    --------------------------------------------------------------------------
    Bcl   dictionary of model coefficients for each class
    res   (optional) Model residuals: (same shape as data), note that
          the first P (depending on the class) residuals are invalid.
    Ccl   (optional) dictionary of residual covariances for each class
    '''
    
    data = np.atleast_3d(data)
    cl = np.asarray(cl)
    
    labels = np.unique(cl)
    
    if cl.size != data.shape[2]:
        raise AttributeError('cl must contain a class label for each trial (expected size %d, but got %d).'%(data.shape[2], cl.size))
    
    if isinstance(P, numbers.Number):
        P = dict.fromkeys(labels, P)
    else:
        try:
            assert(set(labels)==set(P.keys()))
        except:
            raise AttributeError('Model order P must be either a scalar number, or a dictionary containing a key for each unique label in cl.')
    
    Bcl, Ccl = {}, {}
    res = np.zeros(data.shape)
    for c in labels:
        X = data[:,:,cl==c]
        B = fit( X, P[c], delta )            
        Bcl[c] = B
    
        if return_residuals or return_covariance:
            r = X - predict( X, B )
            
        if return_residuals:
             res[:,:,cl==c] = r
            
        if return_covariance:
            Ccl[c] = np.cov(datatools.cat_trials(r), rowvar=False)    
    
    if return_residuals or return_covariance:
        result = [Bcl]
    else:
        return Bcl
        
    if return_residuals:
        result.append(res)
        
    if return_covariance:
        result.append(Ccl)
        
    return tuple(result)
    
    
    
def simulate( L, B, noisefunc=None ):
    '''
    simulate( L, B )
    simulate( L, B, noisefunc )
    
    Simulate vector autoregressive (VAR) model with optional noise generating function.
    
    Note on the arrangement of model coefficients:
        B is of shape M, M*P, with sub matrices arranged as follows:
            b_00 b_01 ... b_0M
            b_10 b_11 ... b_1M
            .... ....     ....
            b_M0 b_M1 ... b_MM
        Each sub matrix b_ij is a column vector of length P that contains the
        filter coefficients from channel j (source) to channel i (sink).
    
    Parameters     Default  Shape   Description
    --------------------------------------------------------------------------
    L              :      : 1     : Number of samples to generate
                   :      : 2     : L[0]: number of samples, L[1]: number of trials
    B              :      : M,M*P : Model coefficients
    noisefunc      : None :       : callback function that takes no parameter and returns M values
    
    Output           Shape   Description
    --------------------------------------------------------------------------
    data           : L,M,T : 3D data matrix
    '''
    B = np.atleast_2d(B)
    (M,N) = np.shape( B )
    P = int(N / M)
    
    try:
        (L,T) = L
    except TypeError:
        T = 1
        
    if noisefunc==None:
        noisefunc = partial( np.random.normal, size=(1,M) )
        
    N = L + 10 * P;
    
    y = np.zeros((N,M,T))
    
    for t in range(T):
        for n in range(P):
            y[n,:,t] = noisefunc()
        for n in range(P,N):
            y[n,:,t] = noisefunc()
            for p in range(1,P+1):
                y[n,:,t] += np.dot( B[:,(p-1)::P], y[n-p,:,t] )
                
    return y[10*P:,:,:]
    
    
    
def predict( data, B ):
    '''
    predict( data, B )
    
    Predict samples from actual data using VAR model coefficients B.
    
    Note that the model requires P past samples for prediction. Thus, the first
    P samples are invalid and set to 0.
    
    Note on the arrangement of model coefficients:
        B is of shape M, M*P, with sub matrices arranged as follows:
            b_00 b_01 ... b_0M
            b_10 b_11 ... b_1M
            .... ....     ....
            b_M0 b_M1 ... b_MM
        Each sub matrix b_ij is a column vector of length P that contains the
        filter coefficients from channel j (source) to channel i (sink).
    
    Parameters     Default  Shape   Description
    --------------------------------------------------------------------------
    data           :      : N,M,T : 3d data matrix (N samples, M signals, T trials)
                          : N,M   : 2d data matrix (N samples, M signals)
    B              :      : M,M*P : Model coefficients
    
    Output           Shape   Description
    --------------------------------------------------------------------------
    predicted      : L,M,T : 3D data matrix
    '''
    data = np.atleast_3d(data)
    (L,M,T) = data.shape
    
    P = int(np.shape(B)[1] / M)

    y = np.zeros(np.shape(data))
    for p in range(1,P+1):
        Bp = B[:,(p-1)::P]
        for n in range(P,L):
            y[n,:,:] += np.dot( Bp, data[n-p,:,:] )
    return y
    

    
def optimize_delta_bisection( data, P, xvschema=lambda t,T: defaults.xvschema(t,T), skipstep=1 ):
    '''
    optimize_delta_bisection( data, P )
    optimize_delta_bisection( data, P, xvschema )
    optimize_delta_bisection( data, P, xvschema, skipstep )
    
    Use the bisection method to find optimal regularization parameter delta.
    
    Parameters     Default  Shape   Description
    --------------------------------------------------------------------------
    data           :      : N,M,T : 3d data matrix (N samples, M signals, T trials)
                          : N,M   : 2d data matrix (N samples, M signals)
    P              :      :       : Model order
    xvschema       :      : func  : Function to generate training and testing set.
                                    See xvschema module.
    skipstep       : 1    : 1     : Higher values speed up the calculation but
                                    cause higher variance in cost function which
                                    will result in less accurate results.
    
    Output           Shape   Description
    --------------------------------------------------------------------------
    delta         : 1     : Optimal regularization parameter
    '''
    data = np.atleast_3d(data)
    (L,M,T) = data.shape
    
    underdetermined = _is_underdetermined(L, M, 1, P)
    
    MAXSTEPS = 10
    MININTERVAL = 1
    MAXDELTA = 1e50
    
    a = -10
    b = 10
    
    transform = lambda x: np.sqrt(np.exp(x))
    
    (Ja,Ka) = _msge_with_gradient(data, P, transform(a), underdetermined, xvschema, skipstep)
    (Jb,Kb) = _msge_with_gradient(data, P, transform(b), underdetermined, xvschema, skipstep)
    
    # before starting the real bisection, make sure the interval actually contains 0
    while np.sign(Ka) == np.sign(Kb):
        print( 'Bisection initial interval (%f,%f) does not contain zero. New interval: (%f,%f)'%(a,b,a*2,b*2) )
        a *= 2
        b *= 2
        (Jb,Kb) = _msge_with_gradient(data, P, transform(b), underdetermined, xvschema, skipstep)
        
        if transform(b) >= MAXDELTA:
            print( 'Bisection: could not find initial interval.')
            print( ' ********* Delta set to zero! ************ ')
            return 0
    
    nsteps = 0
    
    K = np.inf
    sqrtei = np.sqrt(np.exp([a,b]))
    while nsteps < MAXSTEPS:
        
        # point where the line between a and b crosses zero
        # this is not very stable!
        #c = a + (b-a) * np.abs(Ka) / np.abs(Kb-Ka)
        c = (a+b)/2
        (J,K) = _msge_with_gradient(data, P, transform(c), underdetermined, xvschema, skipstep)
        if np.sign(K) == np.sign(Ka):
            a, Ka = c, K
        else:
            b, Kb = c, K
        
        nsteps += 1
        sqrtei = transform([a,b])
        #print('%d Bisection Interval: %f - %f, (projected: %f)'%(nsteps, sqrtei[0], sqrtei[1], transform(a + (b-a) * np.abs(Ka) / np.abs(Kb-Ka))))
        print('%d Bisection Interval: %f - %f, (projected: %f)'%(nsteps, transform(a), transform(b), transform(a + (b-a) * np.abs(Ka) / np.abs(Kb-Ka))))
    
    delta = transform( a + (b-a) * np.abs(Ka) / np.abs(Kb-Ka) )
    print('Final point: %f'%delta)
    return delta



def optimize_delta_gradientdescent( data, P, skipstep=1, xvschema=lambda t,T: defaults.xvschema(t,T) ):
    '''
    optimize_delta_bisection( data, P )
    optimize_delta_bisection( data, P, xvschema )
    optimize_delta_bisection( data, P, xvschema, skipstep )
    
    Use gradient descent to find optimal regularization parameter delta.
    Stable but slow. Use optimize_delta_bisection instead.
    
    Parameters     Default  Shape   Description
    --------------------------------------------------------------------------
    data           :      : N,M,T : 3d data matrix (N samples, M signals, T trials)
                          : N,M   : 2d data matrix (N samples, M signals)
    P              :      :       : Model order
    xvschema       :      : func  : Function to generate training and testing set.
                                    See xvschema module.
    skipstep       : 1    : 1     : Higher values speed up the calculation but
                                    cause higher variance in cost function which
                                    will result in less accurate results.
    
    Output           Shape   Description
    --------------------------------------------------------------------------
    delta         : 1     : Optimal regularization parameter
    '''
    data = np.atleast_3d(data)
    (L,M,T) = data.shape
    
    underdetermined = _is_underdetermined(L, M, 1, P)
    
    K = np.inf
    delta = 1
    step = np.inf
    last_J = np.inf
    J = np.inf
    nsteps = 0
    #while not(last_J-J < 1e-10 and step < 1e-5) and nsteps < 100:        
    while abs(K) > 1e-5:
        last_J = J
        (J,K) = _msge_with_gradient(data, P, delta, underdetermined, xvschema, skipstep)
        
        step = -K * 1.0
                
        delta += step
        
        nsteps += 1
        print('%d Gradient Descent: %f'%(nsteps, K))
    
    return np.sqrt(np.exp(delta))
    
    
 ############################################################################    
    
    
def _msge_crossvalidated( data, P, delta, xvschema, skipstep):
    '''leave-one-trial-out cross validation of VAR prediction error'''
    data = np.atleast_3d(data)
    (L,M,T) = np.shape( data )    
    assert(T>1)
    
    msge = []   # mean squared generalization error
    for t in range(0,T,skipstep):        
        trainset, testset = xvschema(t,T)
        
        traindata = np.atleast_3d(data[:,:,trainset])
        testdata = np.atleast_3d(data[:,:,testset])
        
        B = fit(traindata, P, delta)
        r = (testdata - predict(testdata, B))[P:,:,:]
        
        msge.append( np.mean(r**2) )
        
    return np.mean(msge)
    
def _is_underdetermined( L, M, T, P ):
    N = (L-P)*T     # number of linear relations
    return N < M*P
    
def _msge_with_gradient_underdetermined( data, P, delta, xvschema, skipstep):
    (L,M,T) = data.shape
    
    J, K = 0, 0
    NT = np.ceil(T/skipstep)
    for t in range(0,T,skipstep):        
        trainset, testset = xvschema(t,T)
        
        (A,b) = __construct_eqns( np.atleast_3d(data[:,:,trainset]), P )
        (B,c) = __construct_eqns( np.atleast_3d(data[:,:,testset]), P )
        
        D = np.linalg.inv(np.eye(A.shape[0])*delta**2 + A.dot(A.transpose()))
            
        BB = B.transpose().dot(B)

        bD = b.transpose().dot(D)
        bDD = bD.dot(D)
        bDA = bD.dot(A)
        bDDA = bDD.dot(A)
        bDABB = bDA.dot(BB)
        cB = c.transpose().dot(B)
        
        J += np.sum(bDABB*bDA - 2*bDA*cB) + np.sum(c**2)
        K += np.sum(bDDA*cB - bDABB*bDDA) * 4 * delta
            
    return (J / (NT*c.size), K / (NT*c.size))
    
def _msge_with_gradient_overdetermined( data, P, delta, xvschema, skipstep):
    (L,M,T) = data.shape

    J, K = 0, 0
    NT = np.ceil(T/skipstep)
    for t in range(0,T,skipstep):        
        #print(t,drange)
        trainset, testset = xvschema(t,T)
        
        (A,b) = __construct_eqns( np.atleast_3d(data[:,:,trainset]), P )
        (B,c) = __construct_eqns( np.atleast_3d(data[:,:,testset]), P )

        D = sp.linalg.inv(np.eye(A.shape[1])*delta**2 + A.transpose().dot(A), overwrite_a=True, check_finite=False)
	
        bA = b.transpose().dot(A)
        cB = c.transpose().dot(B)
        bAD = bA.dot(D)
        bADD = bAD.dot(D)
        bADBB = bAD.dot(B.transpose().dot(B))
           
        J += np.sum(bADBB*bAD - 2*bAD*cB) + np.sum(c**2)
        K += np.sum(bADD*cB - bADBB*bADD) * 4 * delta
            
    return (J / (NT*c.size), K / (NT*c.size))
    
def _msge_with_gradient( data, P, delta, underdetermined, xvschema, skipstep ):
    data = np.atleast_3d(data)
    (L,M,T) = data.shape
    assert(T>1)
    
    if underdetermined==None:
        underdetermined = _is_underdetermined(L, M, T, P)

    if underdetermined:
        return _msge_with_gradient_underdetermined( data, P, delta, xvschema, skipstep )
    else:
        return _msge_with_gradient_overdetermined( data, P, delta, xvschema, skipstep )

    
    
    
def __construct_eqns( data, P ):
    '''construct VAR equation system'''
    (L,M,T) = np.shape( data )
    N = (L-P)*T     # number of linear relations
    # construct matrix X (predictor variables)
    X = np.zeros( (N, M*P) )
    for m in range(M):
        for p in range(1,P+1):
            X[:,m*P+p-1] = np.reshape( data[P-p:-p, m, :], N )
            
    # construct vectors yi (response variables for each channel i)
    y = np.zeros( (N, M) );
    for i in range(M):
        y[:,i] = np.reshape( data[P:, i, :], N );
            
    return X, y   
    
def __construct_eqns_RLS( data, P, sqrtdelta ):
    '''construct VAR equation system with RLS constraint'''
    (L,M,T) = np.shape( data )
    N = (L-P)*T     # number of linear relations
    # construct matrix X (predictor variables)
    X = np.zeros( (N + M*P, M*P) )
    for m in range(M):
        for p in range(1,P+1):
            X[:N,m*P+p-1] = np.reshape( data[P-p:-p, m, :], N )
    np.fill_diagonal(X[N:,:], sqrtdelta)
            
    # construct vectors yi (response variables for each channel i)
    y = np.zeros( (N + M*P, M) );
    for i in range(M):
        y[:N,i] = np.reshape( data[P:, i, :], N );
            
    return X, y
    