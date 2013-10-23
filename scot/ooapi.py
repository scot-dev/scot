# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

""" Object oriented API to SCoT """

import numpy as np
from .varica import mvarica, cspvarica
from .plainica import plainica
from .datatools import dot_special
from .connectivity import Connectivity
from . import plotting
from . import var
from eegtopo.topoplot import Topoplot


class Workspace:
    
    def __init__(self, var_order, var_delta=None, locations=None, reducedim=0.99, nfft=512, fs=2, backend=None):
        self.data_ = None
        self.cl_ = None
        self.fs_ = fs
        self.time_offset_ = 0
        self.unmixing_ = None
        self.mixing_ = None
        self.activations_ = None
        self.var_model_ = None
        self.var_cov_ = None
        self.var_order_ = var_order
        self.var_delta_ = var_delta
        self.connectivity_ = None
        self.locations_ = locations
        self.reducedim_ = reducedim
        self.nfft_ = nfft
        self.backend_ = backend
        
        self.topo_ = None
        self.mixmaps_ = []
        self.unmixmaps_ = []
        
    def __str__(self):
        
        if self.data_ is not None:
            data = '%d samples, %d channels, %d trials'%self.data_.shape
        else:
            data = 'None'
            
            
        if self.cl_ is not None:
            cl = str(np.unique(self.cl_))
        else:
            cl = 'None'

        if self.unmixing_ is not None:
            sources = str(self.unmixing_.shape[1])
        else:
            sources = 'None'
            
        if self.var_model_ is None:
            var = 'None'
        elif isinstance(self.var_model_, dict):
            var = str(len(self.var_model_))
        else:
            var = '1'
        
        s = 'SCoT(var_order = %d):\n'%self.var_order_
        s += '  Data      : ' + data + '\n'
        s += '  Classes   : ' + cl + '\n'
        s += '  Sources   : ' + sources + '\n'
        s += '  VAR models: ' + var + '\n'
        
        return s
    
    def setData(self, data, cl=None, time_offset=0):
        self.data_ = np.atleast_3d(data)
        self.cl_ = cl
        self.time_offset_ = time_offset
        self.var_model_ = None
        self.var_cov_ = None
        self.connectivity_ = None
        
        if self.unmixing_ != None:
            self.activations_ = dot_special(self.data_, self.unmixing_)
    
    def doICA(self):
        if self.data_ == None:
            raise RuntimeError("ICA requires data to be set")
        result = plainica(X=self.data_, reducedim=self.reducedim_, backend=self.backend_)
        self.mixing_ = result.mixing
        self.unmixing_ = result.unmixing
        self.activations_ = dot_special(self.data_, self.unmixing_)
        self.var_model_ = None
        self.var_cov_ = None
        self.connectivity_ = None
        self.var_delta_ = None
        self.mixmaps_ = []
        self.unmixmaps_ = []
    
    def doMVARICA(self):
        if self.data_ == None:
            raise RuntimeError("MVARICA requires data to be set")
        result = mvarica(X=self.data_, P=self.var_order_, reducedim=self.reducedim_, delta=self.var_delta_, backend=self.backend_)
        self.mixing_ = result.mixing
        self.unmixing_ = result.unmixing
        self.var_model_ = result.B
        self.var_cov_ = result.C
        self.var_delta_ = result.delta
        self.connectivity_ = Connectivity(self.var_model_, self.var_cov_, self.nfft_)
        self.activations_ = dot_special(self.data_, self.unmixing_)
        self.mixmaps_ = []
        self.unmixmaps_ = []
    
    def doCSPVARICA(self):
        if self.data_ == None:
            raise RuntimeError("CSPVARICA requires data to be set")
        if self.cl_ is None:
            raise RuntimeError("CSPVARICA requires class labels")
        result = cspvarica(X=self.data_, cl=self.cl_, P=self.var_order_, reducedim=self.reducedim_, delta=self.var_delta_, backend=self.backend_)
        self.mixing_ = result.mixing
        self.unmixing_ = result.unmixing
        self.var_model_ = result.B
        self.var_cov_ = result.C
        self.var_delta_ = result.delta
        self.connectivity_ = Connectivity(self.var_model_, self.var_cov_, self.nfft_)
        self.activations_ = dot_special(self.data_, self.unmixing_)
        self.mixmaps_ = []
        self.unmixmaps_ = []
        
    def removeSources(self, sources):
        if self.unmixing_ == None or self.mixing_ == None:
            raise RuntimeError("No sources available (run doMVARICA first)")
        self.mixing_ = np.delete(self.mixing_, sources, 0)
        self.unmixing_ = np.delete(self.unmixing_, sources, 1)
        if self.activations_ != None:
            self.activations_ = np.delete(self.activations_, sources, 1)
        self.var_model_ = None
        self.var_cov_ = None
        self.connectivity_ = None
        
    
    def fitVAR(self):
        if self.activations_ == None:
            raise RuntimeError("VAR fitting requires source activations (run doMVARICA first)")
        if self.cl_ == None:
            self.var_model_, self.var_cov_ = var.fit(data=self.activations_, P=self.var_order_, delta=self.var_delta_, return_covariance=True)
            self.connectivity_ = Connectivity(self.var_model_, self.var_cov_, self.nfft_)
        else:
            self.var_model_, self.var_cov_ = var.fit_multiclass(data=self.activations_, cl=self.cl_, P=self.var_order_, delta=self.var_delta_, return_covariance=True)
            self.connectivity_ = {}
            for c in np.unique(self.cl_):
                self.connectivity_[c] = Connectivity(self.var_model_[c], self.var_cov_[c], self.nfft_)

    def optimizeRegularization(self, xvschema, skipstep=1):
        if self.activations_ == None:
            raise RuntimeError("VAR fitting requires source activations (run doMVARICA first)")
            
        self.var_delta_ = var.optimize_delta_bisection(data=self.activations_, P=self.var_order_, xvschema=xvschema, skipstep=skipstep)
                
    
    def getConnectivity(self, measure):
        if self.connectivity_ == None:
            raise RuntimeError("Connectivity requires a VAR model (run doMVARICA or fitVAR first)")
        if isinstance(self.connectivity_, dict):
            result = {}
            for c in np.unique(self.cl_):
                result[c] = getattr(self.connectivity_[c], measure)()
            return result
        else:
            return getattr(self.connectivity_, measure)()
    
    def getTFConnectivity(self, measure, winlen, winstep):
        if self.activations_ == None:
            raise RuntimeError("Time/Frequency Connectivity requires activations (call setData after doMVARICA)")
        [N,M,T] = self.activations_.shape
        
        Nstep = (N-winlen)//winstep
        
        if self.cl_ == None:
            result = np.zeros((M, M, self.nfft_, Nstep), np.complex64)
            i = 0
            for n in range(0, N-winlen, winstep):
                win = np.arange(winlen) + n
                data = self.activations_[win,:,:]                
                B, C = var.fit(data, P=self.var_order_, delta=self.var_delta_, return_covariance=True)
                con = Connectivity(B, C, self.nfft_)
                result[:,:,:,i] = getattr(con, measure)()
                i += 1
        
        else:
            result = {}
            for c in np.unique(self.cl_):
                result[c] = np.zeros((M, M, self.nfft_, Nstep), np.complex128)
            i = 0
            for n in range(0, N-winlen, winstep):
                win = np.arange(winlen) + n
                data = self.activations_[win,:,:]                
                B, C = var.fit_multiclass(data, cl=self.cl_, P=self.var_order_, delta=self.var_delta_, return_covariance=True)
                for c in np.unique(self.cl_):
                    con = Connectivity(B[c], C[c], self.nfft_)
                    result[c][:,:,:,i] = getattr(con, measure)()
                i += 1
        return result
        
    def preparePlots(self, mixing=False, unmixing=False):
        if self.locations_ == None:
            raise RuntimeError("Need sensor locations for plotting")
            
        if self.topo_ == None:
            self.topo_ = Topoplot( )
            self.topo_.set_locations(self.locations_)
        
        if mixing and not self.mixmaps_:
            self.mixmaps_ = plotting.prepare_topoplots(self.topo_, self.mixing_)
        
        if unmixing and not self.unmixmaps_:
            self.unmixmaps_ = plotting.prepare_topoplots(self.topo_, self.unmixing_.transpose())
                
    def showPlots(self):
        plotting.show_plots( )
    
    def plotSourceTopos(self, global_scale=None):
        """ global_scale:
               None - scales each topo individually
               1-99 - percentile of maximum of all plots
        """
        if self.unmixing_ == None and self.mixing_ == None:
            raise RuntimeError("No sources available (run doMVARICA first)")
            
        self.preparePlots(True, True)

        plotting.plot_sources(self.topo_, self.mixmaps_, self.unmixmaps_, global_scale)
    
    def plotConnectivity(self, measure, freq_range=[-np.inf, np.inf]):
        fig = None        
        self.preparePlots(True, False)        
        if isinstance(self.connectivity_, dict):            
            for c in np.unique(self.cl_):
                cm = getattr(self.connectivity_[c], measure)()
                fig = plotting.plot_connectivity_spectrum(cm, fs=self.fs_, freq_range=freq_range, topo=self.topo_, topomaps=self.mixmaps_, fig=fig)
        else:
            cm = getattr(self.connectivity_, measure)()
            fig = plotting.plot_connectivity_spectrum(cm, fs=self.fs_, freq_range=freq_range, topo=self.topo_, topomaps=self.mixmaps_)
        return fig
    
    def plotTFConnectivity(self, measure, winlen, winstep, freq_range=[-np.inf, np.inf], ignore_diagonal=True):
        fig = None        
        
        t0 = 0.5*winlen/self.fs_ + self.time_offset_
        t1 = self.data_.shape[0]/self.fs_ - 0.5*winlen/self.fs_ + self.time_offset_
        
        self.preparePlots(True, False)
        tfc = self.getTFConnectivity(measure, winlen, winstep)
        
        if isinstance(tfc, dict):
            ncl = np.unique(self.cl_).size
            Y = np.floor(np.sqrt(ncl))
            X = np.ceil(ncl/Y)
            lowest, highest = np.inf, -np.inf
            for c in np.unique(self.cl_):
                tfc[c] = self._cleanMeasure(measure, tfc[c])
                if ignore_diagonal:
                    for m in range(tfc[c].shape[0]):
                        tfc[c][m,m,:,:] = 0;
                highest = max(highest, np.max(tfc[c]))
                lowest = min(lowest, np.min(tfc[c]))
                
            fig = {}
            for c in np.unique(self.cl_):
                fig[c] = plotting.plot_connectivity_timespectrum(tfc[c], fs=self.fs_, crange=[lowest, highest], freq_range=freq_range, time_range=[t0, t1], topo=self.topo_, topomaps=self.mixmaps_)
                
        else:
            tfc = self._cleanMeasure(measure, tfc)
            if ignore_diagonal:
                for m in range(tfc.shape[0]):
                    tfc[m,m,:,:] = 0;
            fig = plotting.plot_connectivity_timespectrum(tfc, fs=self.fs_, crange=[np.min(tfc), np.max(tfc)], freq_range=freq_range, time_range=[t0, t1], topo=self.topo_, topomaps=self.mixmaps_)
        return fig
            
    def _cleanMeasure(self, measure, A):
        if measure in ['A', 'H', 'COH', 'pCOH']:
            return np.abs(A)
        elif measure in ['S', 'G']:
            return np.log(np.abs(A))
        else:
            return np.real(A)
