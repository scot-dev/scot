# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

""" Object oriented API to SCoT """

import numpy as np
from .varica import mvarica
from .datatools import dot_special
from .connectivity import Connectivity
from . import var

class SCoT:
    
    def __init__(self, var_order, backend=None):
        self.data_ = None
        self.cl_ = None
        self.unmixing_ = None
        self.mixing_ = None
        self.activations_ = None
        self.var_model_ = None
        self.var_cov_ = None
        self.var_order_ = var_order
        self.var_delta_ = None
        self.connectivity_ = None
        
        self.backend_ = backend
        
        self.reducedim_ = 0.99
        self.nfft_ = 512
    
    def setData(self, data, cl=None):
        self.data_ = np.atleast_3d(data)
        self.cl_ = cl
        self.var_model_ = None
        self.var_cov_ = None
        self.connectivity_ = None
        
        if self.unmixing_ != None:
            self.activations_ = dot_special(self.data_, self.unmixing_)
    
    def doMVARICA(self):
        if self.data_ == None:
            raise RuntimeError("MVARICA requires data to be set")
        if self.reducedim_ < 1:
            rv = self.reducedim_
            nc = None
        else:
            rv = None
            nc = self.reducedim_
        result = mvarica(X=self.data_, P=self.var_order_, retain_variance=rv, numcomp=nc, delta=self.var_delta_, backend=self.backend_)
        self.mixing_ = result.mixing
        self.unmixing_ = result.unmixing
        self.var_model_ = result.B
        self.var_cov_ = result.C
        self.var_delta_ = result.delta
        self.connectivity_ = Connectivity(self.var_model_, self.var_cov_, self.nfft_)
    
    def fitVAR(self):
        if self.activations_ == None:
            raise RuntimeError("VAR fitting requires activations (call setData after doMVARICA)")
        if self.cl_ == None:
            self.var_model_, self.var_cov_ = var.fit(data=self.data_, P=self.var_order_, delta=self.var_delta_, return_covariance=True)
            self.connectivity_ = Connectivity(self.var_model_, self.var_cov_, self.nfft_)
        else:
            self.var_model_, self.var_cov_ = var.fit_multiclass(data=self.data_, cl=self.cl_, P=self.var_order_, delta=self.var_delta_, return_covariance=True)
            self.connectivity_ = {}
            for c in np.unique(self.cl_):
                self.connectivity_[c] = Connectivity(self.var_model_[c], self.var_cov_[c], self.nfft_)
    
    def getConnectivity(self, measure):
        if self.connectivity_ == None:
            raise RuntimeError("Connectivity requires a VAR model (run doMVARICA or fitVAR first)")
        if isinstance(self.connectivity_, dict):
            result = {}
            for c in np.unique(self.cl_):
                result[c] = getattr(self.connectivity_[c], measure)
            return result
        else:
            return getattr(self.connectivity_, measure)
    
    def getTFConnectivity(self, measure, winlen, winstep):
        if self.activations_ == None:
            raise RuntimeError("Time/Frequency Connectivity requires activations (call setData after doMVARICA)")
        [N,M,T] = self.activations_.shape
        
        Nstep = (N-winlen)//winstep
        
        if self.cl_ == None:
            result = np.zeros((M, M, Nstep, self.nfft_), np.complex64)
            i = 0
            for n in range(0, N-winlen, winstep):
                win = np.arange(winlen) + n
                data = self.activations_[win,:,:]                
                B, C = var.fit(data, P=self.var_order_, delta=self.var_delta_, return_covariance=True)
                con = Connectivity(B, C, self.nfft_)
                result[:,:,i,:] = getattr(con, measure)()
                i += 1
        
        else:
            result = {}
            for c in np.unique(self.cl_):
                result[c] = np.zeros((M, M, Nstep, self.nfft_), np.complex64)
            i = 0
            for n in range(0, N-winlen, winstep):
                win = np.arange(winlen) + n
                data = self.activations_[win,:,:]                
                B, C = var.fit_multiclass(data, cl=self.cl_, P=self.var_order_, delta=self.var_delta_, return_covariance=True)
                for c in np.unique(self.cl_):
                    con = Connectivity(B[c], C[c], self.nfft_)
                    result[c][:,:,i,:] = getattr(con, measure)()
                i += 1
        return result
    
    def plotComponents(self):
        if self.unmixing_ == None and self.mixing_ == None:
            raise RuntimeError("No components available (run doMVARICA first)")
    
    def plotConnectivity(self, measure):
        pass