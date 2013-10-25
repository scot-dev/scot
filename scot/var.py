# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

""" vector autoregressive (VAR) model implementation"""

import numbers
import numpy as np
from functools import partial
from . import datatools
from . import xvschema as xv

class Defaults:
    xvschema = xv.multitrial

def fit( data, p, delta=None, return_residuals=False, return_covariance=False ):
    """
    fit( data, p )
    fit( data, p, delta )

    Fit least squares estimate of vector autoregressive (VAR) model with
    order p to the data.

    If sqrtdelta is provited and nonzero, the least squares estimation is
    regularized with ridge regression.

    Parameters       Default  Shape   Description
    --------------------------------------------------------------------------
    data             :      : n,m,T : 3d data matrix (n samples, m signals, T trials)
                            : n,m   : 2d data matrix (n samples, m signals)
    p                :      :       : Model order
    delta            : None :       : regularization parameter
    return_residuals : False :      : if True, also return model residuals
    return_covariance: False :      : if True, also return covariance

    Output
    --------------------------------------------------------------------------
    b   Model coefficients: [b_0, b_1, ... b_p], each sub matrix b_k is of size m*m
    res (optional) Model residuals: (same shape as data), note that
        the first p residuals are invalid.
    c   (optional) Covariance of residuals

    Note on the arrangement of model coefficients:
        b is of shape m, m*p, with sub matrices arranged as follows:
            b_00 b_01 ... b_0m
            b_10 b_11 ... b_1m
            .... ....     ....
            b_m0 b_m1 ... b_mm
        Each sub matrix b_ij is a column vector of length p that contains the
        filter coefficients from channel j (source) to channel i (sink).
    """
    data = np.atleast_3d(data)
    
    if delta == 0 or delta is None:
        # normal least squares
        (x,y) = __construct_eqns( data, p )
    else:    
        # regularized least squares
        (x,y) = __construct_eqns_rls( data, p, delta )
    
    (b, res, rank, s) = np.linalg.lstsq( x, y )
    b = b.transpose()
    
    if return_residuals or return_covariance:
        result = [b]
        res = data - predict( data, b )
    else:
        return b
        
    if return_residuals:
        result.append(res)
        
    if return_covariance:
        c = np.cov(datatools.cat_trials(res), rowvar=False)
        result.append(c)
        
    return tuple(result)
   
   
    
def fit_multiclass( data, cl, p, delta=None, return_residuals=False, return_covariance=False ):
    """
    fit_multiclass( data, cl, p )
    fit_multiclass( data, cl, p, delta )

    Fits a separate autoregressive model for each class.

    If sqrtdelta is provited and nonzero, the least squares estimation is
    regularized with ridge regression.

    Parameters     Default  Shape   Description
    --------------------------------------------------------------------------
    data             :      : n,m,T : 3d data matrix (n samples, m signals, T trials)
    cl               :      : T     : class label for each trial
    p                :      :       : Model order can be scalar (same model order for each class)
                                      or a dictionary that contains a key for each unique class label
                                      to specify the model order for each class individually.
    delta            : None :       : regularization parameter
    return_residuals : False :      : if True, also return model residuals
    return_covariance: False :      : if True, also return dictionary of covariances

    Output
    --------------------------------------------------------------------------
    bcl   dictionary of model coefficients for each class
    res   (optional) Model residuals: (same shape as data), note that
          the first p (depending on the class) residuals are invalid.
    ccl   (optional) dictionary of residual covariances for each class
    """
    
    data = np.atleast_3d(data)
    cl = np.asarray(cl)
    
    labels = np.unique(cl)
    
    if cl.size != data.shape[2]:
        raise AttributeError('cl must contain a class label for each trial (expected size %d, but got %d).'%(data.shape[2], cl.size))
    
    if isinstance(p, numbers.Number):
        p = dict.fromkeys(labels, p)
    else:
        try:
            assert(set(labels)==set(p.keys()))
        except:
            raise AttributeError('Model order p must be either a scalar number, or a dictionary containing a key for each unique label in cl.')
    
    bcl, ccl = {}, {}
    res = np.zeros(data.shape)
    for c in labels:
        x = data[:,:,cl==c]
        b = fit( x, p[c], delta )
        bcl[c] = b
    
        if return_residuals or return_covariance:
            r = x - predict( x, b )
            
        if return_residuals:
             res[:,:,cl==c] = r
            
        if return_covariance:
            ccl[c] = np.cov(datatools.cat_trials(r), rowvar=False)

    result = []

    if return_residuals or return_covariance:
        result.append(bcl)
    else:
        return bcl
        
    if return_residuals:
        result.append(res)
        
    if return_covariance:
        result.append(ccl)
        
    return tuple(result)
    
    
    
def simulate( l, b, noisefunc=None ):
    """
    simulate( l, b )
    simulate( l, b, noisefunc )

    Simulate vector autoregressive (VAR) model with optional noise generating function.

    Note on the arrangement of model coefficients:
        b is of shape m, m*p, with sub matrices arranged as follows:
            b_00 b_01 ... b_0m
            b_10 b_11 ... b_1m
            .... ....     ....
            b_m0 b_m1 ... b_mm
        Each sub matrix b_ij is a column vector of length p that contains the
        filter coefficients from channel j (source) to channel i (sink).

    Parameters     Default  Shape   Description
    --------------------------------------------------------------------------
    l              :      : 1     : Number of samples to generate
                   :      : 2     : l[0]: number of samples, l[1]: number of trials
    b              :      : m,m*p : Model coefficients
    noisefunc      : None :       : callback function that takes no parameter and returns m values

    Output           Shape   Description
    --------------------------------------------------------------------------
    data           : l,m,t : 3D data matrix
    """
    b = np.atleast_2d(b)
    (m,n) = np.shape( b )
    p = int(n / m)
    
    try:
        (l,t) = l
    except TypeError:
        t = 1
        
    if noisefunc is None:
        noisefunc = partial( np.random.normal, size=(1,m) )
        
    n = l + 10 * p

    y = np.zeros((n,m,t))
    
    for s in range(t):
        for i in range(p):
            y[i,:,s] = noisefunc()
        for i in range(p,n):
            y[i,:,s] = noisefunc()
            for k in range(1,p+1):
                y[i,:,s] += np.dot( b[:,(k-1)::p], y[i-k,:,s] )
                
    return y[10*p:,:,:]
    
    
    
def predict( data, b ):
    """
    predict( data, b )

    Predict samples from actual data using VAR model coefficients b.

    Note that the model requires p past samples for prediction. Thus, the first
    p samples are invalid and set to 0.

    Note on the arrangement of model coefficients:
        b is of shape m, m*p, with sub matrices arranged as follows:
            b_00 b_01 ... b_0m
            b_10 b_11 ... b_1m
            .... ....     ....
            b_m0 b_m1 ... b_mm
        Each sub matrix b_ij is a column vector of length p that contains the
        filter coefficients from channel j (source) to channel i (sink).

    Parameters     Default  Shape   Description
    --------------------------------------------------------------------------
    data           :      : n,m,t : 3d data matrix (n samples, m signals, t trials)
                          : n,m   : 2d data matrix (n samples, m signals)
    b              :      : m,m*p : Model coefficients

    Output           Shape   Description
    --------------------------------------------------------------------------
    predicted      : l,m,t : 3D data matrix
    """
    data = np.atleast_3d(data)
    (l,m,t) = data.shape
    
    p = int(np.shape(b)[1] / m)

    y = np.zeros(np.shape(data))
    for k in range(1,p+1):
        bp = b[:,(k-1)::p]
        for n in range(p,l):
            y[n,:,:] += np.dot( bp, data[n-k,:,:] )
    return y
    

    
def optimize_delta_bisection( data, p, xvschema=lambda t,nt: Defaults.xvschema(t,nt), skipstep=1 ):
    """
    optimize_delta_bisection( data, p )
    optimize_delta_bisection( data, p, xvschema )
    optimize_delta_bisection( data, p, xvschema, skipstep )

    Use the bisection method to find optimal regularization parameter delta.

    Parameters     Default  Shape   Description
    --------------------------------------------------------------------------
    data           :      : n,m,t : 3d data matrix (n samples, m signals, t trials)
                          : n,m   : 2d data matrix (n samples, m signals)
    p              :      :       : Model order
    xvschema       :      : func  : Function to generate training and testing set.
                                    See xvschema module.
    skipstep       : 1    : 1     : Higher values speed up the calculation but
                                    cause higher variance in cost function which
                                    will result in less accurate results.

    Output           Shape   Description
    --------------------------------------------------------------------------
    delta         : 1     : Optimal regularization parameter
    """
    data = np.atleast_3d(data)
    (l,m,t) = data.shape
    
    underdetermined = _is_underdetermined(l, m, 1, p)
    
    maxsteps = 10
    maxdelta = 1e50
    
    a = -10
    b = 10
    
    transform = lambda x: np.sqrt(np.exp(x))
    
    (ja,ka) = _msge_with_gradient(data, p, transform(a), underdetermined, xvschema, skipstep)
    (jb,kb) = _msge_with_gradient(data, p, transform(b), underdetermined, xvschema, skipstep)
    
    # before starting the real bisection, make sure the interval actually contains 0
    while np.sign(ka) == np.sign(kb):
        print( 'Bisection initial interval (%f,%f) does not contain zero. New interval: (%f,%f)'%(a,b,a*2,b*2) )
        a *= 2
        b *= 2
        (jb,kb) = _msge_with_gradient(data, p, transform(b), underdetermined, xvschema, skipstep)
        
        if transform(b) >= maxdelta:
            print( 'Bisection: could not find initial interval.')
            print( ' ********* Delta set to zero! ************ ')
            return 0
    
    nsteps = 0
    
    while nsteps < maxsteps:
        
        # point where the line between a and b crosses zero
        # this is not very stable!
        #c = a + (b-a) * np.abs(ka) / np.abs(kb-ka)
        c = (a+b)/2
        (j,k) = _msge_with_gradient(data, p, transform(c), underdetermined, xvschema, skipstep)
        if np.sign(k) == np.sign(ka):
            a, ka = c, k
        else:
            b, kb = c, k
        
        nsteps += 1
        tmp = transform([a, b, a + (b-a) * np.abs(ka) / np.abs(kb-ka)])
        print('%d Bisection Interval: %f - %f, (projected: %f)'%(nsteps, tmp[0], tmp[1], tmp[2]))
    
    delta = transform( a + (b-a) * np.abs(ka) / np.abs(kb-ka) )
    print('Final point: %f'%delta)
    return delta



def optimize_delta_gradientdescent( data, p, skipstep=1, xvschema=lambda t,nt: Defaults.xvschema(t,nt) ):
    """
    optimize_delta_bisection( data, p )
    optimize_delta_bisection( data, p, xvschema )
    optimize_delta_bisection( data, p, xvschema, skipstep )

    Use gradient descent to find optimal regularization parameter delta.
    Stable but slow. Use optimize_delta_bisection instead.

    Parameters     Default  Shape   Description
    --------------------------------------------------------------------------
    data           :      : n,m,t : 3d data matrix (n samples, m signals, t trials)
                          : n,m   : 2d data matrix (n samples, m signals)
    p              :      :       : Model order
    xvschema       :      : func  : Function to generate training and testing set.
                                    See xvschema module.
    skipstep       : 1    : 1     : Higher values speed up the calculation but
                                    cause higher variance in cost function which
                                    will result in less accurate results.

    Output           Shape   Description
    --------------------------------------------------------------------------
    delta         : 1     : Optimal regularization parameter
    """
    data = np.atleast_3d(data)
    (l,m,t) = data.shape
    
    underdetermined = _is_underdetermined(l, m, 1, p)
    
    k = np.inf
    delta = 1
    nsteps = 0
    while abs(k) > 1e-5:
        (j, k) = _msge_with_gradient(data, p, delta, underdetermined, xvschema, skipstep)
        
        step = -k * 1.0
                
        delta += step
        
        nsteps += 1
        print('%d Gradient Descent: %f'%(nsteps, k))
    
    return np.sqrt(np.exp(delta))
    
    
 ############################################################################    
    
    
def _msge_crossvalidated( data, p, delta, xvschema, skipstep):
    """leave-one-trial-out cross validation of VAR prediction error"""
    data = np.atleast_3d(data)
    (l,m,t) = np.shape( data )
    assert(t>1)
    
    msge = []   # mean squared generalization error
    for s in range(0,t,skipstep):
        trainset, testset = xvschema(s,t)
        
        traindata = np.atleast_3d(data[:,:,trainset])
        testdata = np.atleast_3d(data[:,:,testset])
        
        b = fit(traindata, p, delta)
        r = (testdata - predict(testdata, b))[p:,:,:]
        
        msge.append( np.mean(r**2) )
        
    return np.mean(msge)
    
def _is_underdetermined( l, m, t, p ):
    n = (l-p)*t     # number of linear relations
    return n < m*p
    
def _msge_with_gradient_underdetermined( data, p, delta, xvschema, skipstep):
    (l,m,t) = data.shape
    d = None
    j, k = 0, 0
    nt = np.ceil(t/skipstep)
    for s in range(0,t,skipstep):
        trainset, testset = xvschema(s,t)
        
        (a,b) = __construct_eqns( np.atleast_3d(data[:,:,trainset]), p )
        (c,d) = __construct_eqns( np.atleast_3d(data[:,:,testset]), p )
        
        e = np.linalg.inv(np.eye(a.shape[0])*delta**2 + a.dot(a.transpose()))
            
        cc = c.transpose().dot(c)

        be = b.transpose().dot(e)
        bee = be.dot(e)
        bea = be.dot(a)
        beea = bee.dot(a)
        beacc = bea.dot(cc)
        dc = d.transpose().dot(c)
        
        j += np.sum(beacc*bea - 2*bea*dc) + np.sum(d**2)
        k += np.sum(beea*dc - beacc*beea) * 4 * delta
            
    return j / (nt*d.size), k / (nt*d.size)
    
def _msge_with_gradient_overdetermined( data, p, delta, xvschema, skipstep):
    (l,m,t) = data.shape
    d = None
    l, k = 0, 0
    nt = np.ceil(t/skipstep)
    for s in range(0,t,skipstep):
        #print(s,drange)
        trainset, testset = xvschema(s,t)
        
        (a,b) = __construct_eqns( np.atleast_3d(data[:,:,trainset]), p )
        (c,d) = __construct_eqns( np.atleast_3d(data[:,:,testset]), p )

        #e = sp.linalg.inv(np.eye(a.shape[1])*delta**2 + a.transpose().dot(a), overwrite_a=True, check_finite=False)
        e = np.linalg.inv(np.eye(a.shape[1])*delta**2 + a.transpose().dot(a))
	
        ba = b.transpose().dot(a)
        dc = d.transpose().dot(c)
        bae = ba.dot(e)
        baee = bae.dot(e)
        baecc = bae.dot(c.transpose().dot(c))
           
        l += np.sum(baecc*bae - 2*bae*dc) + np.sum(d**2)
        k += np.sum(baee*dc - baecc*baee) * 4 * delta
            
    return l / (nt*d.size), k / (nt*d.size)
    
def _msge_with_gradient( data, p, delta, underdetermined, xvschema, skipstep ):
    data = np.atleast_3d(data)
    (l,m,t) = data.shape
    assert(t>1)
    
    if underdetermined is None:
        underdetermined = _is_underdetermined(l, m, t, p)

    if underdetermined:
        return _msge_with_gradient_underdetermined( data, p, delta, xvschema, skipstep )
    else:
        return _msge_with_gradient_overdetermined( data, p, delta, xvschema, skipstep )

    
    
    
def __construct_eqns( data, p ):
    """Construct VAR equation system"""
    (l,m,t) = np.shape( data )
    n = (l-p)*t     # number of linear relations
    # Construct matrix x (predictor variables)
    x = np.zeros( (n, m*p) )
    for i in range(m):
        for k in range(1,p+1):
            x[:,i*p+k-1] = np.reshape( data[p-k:-k, i, :], n )
            
    # Construct vectors yi (response variables for each channel i)
    y = np.zeros( (n, m) )
    for i in range(m):
        y[:,i] = np.reshape( data[p:, i, :], n )

    return x, y
    
def __construct_eqns_rls( data, p, sqrtdelta ):
    """Construct VAR equation system with RLS constraint"""
    (l,m,t) = np.shape( data )
    n = (l-p)*t     # number of linear relations
    # Construct matrix x (predictor variables)
    x = np.zeros( (n + m*p, m*p) )
    for i in range(m):
        for k in range(1,p+1):
            x[:n,i*p+k-1] = np.reshape( data[p-k:-k, i, :], n )
    np.fill_diagonal(x[n:,:], sqrtdelta)
            
    # Construct vectors yi (response variables for each channel i)
    y = np.zeros( (n + m*p, m) )
    for i in range(m):
        y[:n,i] = np.reshape( data[p:, i, :], n )

    return x, y
    