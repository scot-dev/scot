# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

""" Object oriented API to SCoT """

import numpy as np
from .varica import mvarica
from .plainica import plainica
from .datatools import dot_special
from .connectivity import Connectivity
from . import var
from eegtopo.topoplot import Topoplot

try:
    import matplotlib.pyplot as plt
    _have_pyplot = True
except ImportError:
    _have_pyplot = False

class SCoT:
    
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
            
            
        if self.data_ is not None:
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
            for i in range(self.mixing_.shape[0]):
                self.topo_.set_values(self.mixing_[i,:])
                self.topo_.create_map()
                self.mixmaps_.append(self.topo_.get_map())
        
        if unmixing and not self.unmixmaps_:
            for i in range(self.unmixing_.shape[1]):
                self.topo_.set_values(self.unmixing_[:,i])
                self.topo_.create_map()
                self.unmixmaps_.append(self.topo_.get_map())
                
    def showPlots(self):
        plt.show()
    
    def plotSourceTopos(self, global_scale=None):
        """ global_scale:
               None - scales each topo individually
               1-99 - percentile of maximum of all plots
        """
        if not _have_pyplot:
            raise ImportError("matplotlib.pyplot is required for plotting")
        if self.unmixing_ == None and self.mixing_ == None:
            raise RuntimeError("No sources available (run doMVARICA first)")
        self.preparePlots(True, True)
        
        M = self.mixing_.shape[0]
        
        urange, mrange = None, None
        
        if global_scale:        
            tmp = np.asarray(self.unmixmaps_)
            tmp = tmp[np.logical_not(np.isnan(tmp))]     
            umax = np.percentile(np.abs(tmp), global_scale)
            umin = -umax
            urange = [umin,umax]
            
            tmp = np.asarray(self.mixmaps_)
            tmp = tmp[np.logical_not(np.isnan(tmp))]   
            mmax = np.percentile(np.abs(tmp), global_scale)
            mmin = -mmax
            mrange = [umin,umax]
            
        Y = np.floor(np.sqrt(M*3/4))
        X = np.ceil(M/Y)
        
        fig = plt.figure()
        
        axes = []
        for m in range(M):
            axes.append(fig.add_subplot(2*Y, X, m+1))
            h1 = self._plotUnmixing(axes[-1], m, crange=urange)
            axes[-1].set_title(str(m))
            
            axes.append(fig.add_subplot(2*Y, X, M+m+1))
            h1 = self._plotMixing(axes[-1], m, crange=mrange)
            axes[-1].set_title(str(m))
            
        for a in axes:
            a.set_yticks([])
            a.set_xticks([])
            a.set_frame_on(False)
            
        axes[0].set_ylabel('Unmixing weights')
        axes[1].set_ylabel('Scalp projections')
        
        #plt.colorbar(h1, plt.subplot(2, M+1, M+1))
        #plt.colorbar(h2, plt.subplot(2, M+1, 0))
    
    def plotConnectivity(self, measure):
        if not _have_pyplot:
            raise ImportError("matplotlib.pyplot is required for plotting")
        self.preparePlots(True, False)
        fig = plt.figure()
        if isinstance(self.connectivity_, dict):
            for c in np.unique(self.cl_):
                cm = getattr(self.connectivity_[c], measure)()
                self._plotSpectral(fig, cm)
        else:
            cm = getattr(self.connectivity_, measure)()
            self._plotSpectral(fig, cm)
    
    def plotTFConnectivity(self, measure, winlen, winstep):
        if not _have_pyplot:
            raise ImportError("matplotlib.pyplot is required for plotting")
        self.preparePlots(True, False)
        tfc = self.getTFConnectivity(measure, winlen, winstep)
        if isinstance(tfc, dict):
            ncl = np.unique(self.cl_).size
            Y = np.floor(np.sqrt(ncl))
            X = np.ceil(ncl/Y)
            lowest, highest = np.inf, -np.inf
            for c in np.unique(self.cl_):
                tfc[c] = self._cleanMeasure(measure, tfc[c])
                highest = max(highest, np.max(tfc[c]))
                lowest = min(lowest, np.min(tfc[c]))
                
            for c in np.unique(self.cl_):
                fig = plt.figure()
                self._plotTimeFrequency(fig, tfc[c], [lowest, highest], winlen, winstep)
        else:
            fig = plt.figure()
            self._plotTimeFrequency(fig, self._cleanMeasure(measure, tfc, [np.min(tfc), np.max(tfc)]), winlen, winstep)
            
    def _cleanMeasure(self, measure, A):
        if measure in ['A', 'H', 'COH', 'pCOH']:
            return np.abs(A)
        elif measure in ['S', 'G']:
            return np.log(np.abs(A))
        else:
            return np.real(A)
            
    def _plotSpectral(self, fig, A):
        [N,M,F] = A.shape
        freq = np.linspace(0, self.fs_/2, F)

        lowest, highest = np.inf, -np.inf
        
        axes = []
        for n in range(N):
            arow = []
            for m in range(M):
                ax = fig.add_subplot(N, M, m+n*M+1)
                arow.append(ax)
                
                if n == m:
                    self._plotMixing(ax, m)
                    ax.set_yticks([])
                    ax.set_xticks([])
                    ax.set_frame_on(False)
                else:                
                    ax.plot(freq, A[n,m,:])
                    lowest = min(lowest, np.min(A[n,m,:]))
                    highest = max(highest, np.max(A[n,m,:]))
                    ax.set_xlim(0, self.fs_/2)
            axes.append(arow)
            
        for n in range(N):
            for m in range(M):
                if n == m:
                    pass
                else:
                    axes[n][m].set_ylim(lowest, highest)
                    if 0 < n < N-1:
                        axes[n][m].set_xticks([])
                    if 0 < m < M-1:
                        axes[n][m].set_yticks([])                    
            axes[n][0].yaxis.tick_left()
            axes[n][-1].yaxis.tick_right()
            
        for m in range(M):
            axes[0][m].xaxis.tick_top()
            axes[-1][m].xaxis.tick_bottom()
            
        fig.text(0.5, 0.05, 'frequency', horizontalalignment='center')
        fig.text(0.05, 0.5, 'magnitude', horizontalalignment='center', rotation='vertical')
            
    def _plotTimeFrequency(self, fig, A, crange, winlen, winstep):
        [N,M,F,T] = A.shape
        fs = self.fs_
        
        f0, f1 = fs/2, 0
        t0 = 0.5*winlen/fs + self.time_offset_
        t1 = self.data_.shape[0]/fs - 0.5*winlen/fs + self.time_offset_        
        extent = [t0, t1, f0, f1]
        
        axes = []
        for n in range(N):
            arow = []
            for m in range(M):
                ax = fig.add_subplot(N, M, m+n*M+1)
                arow.append(ax)
                
                if n == m:
                    self._plotMixing(ax, m)
                    ax.set_yticks([])
                    ax.set_xticks([])
                    ax.set_frame_on(False)
                else:
                    ax.imshow(A[n,m,:,:], vmin=crange[0], vmax=crange[1], aspect='auto', extent=extent)
                    ax.invert_yaxis()
            axes.append(arow)
            
        for n in range(N):
            for m in range(M):
                if n == m:
                    pass
                else:
                    if 0 < n < N-1:
                        axes[n][m].set_xticks([])
                    if 0 < m < M-1:
                        axes[n][m].set_yticks([])                    
            axes[n][0].yaxis.tick_left()
            axes[n][-1].yaxis.tick_right()
            
        for m in range(M):
            axes[0][m].xaxis.tick_top()
            axes[-1][m].xaxis.tick_bottom()
            
        fig.text(0.5, 0.05, 'time', horizontalalignment='center')
        fig.text(0.05, 0.5, 'frequency', horizontalalignment='center', rotation='vertical')
        
    def _plotMixing(self, axis, idx, crange=None):
        self.topo_.set_map(self.mixmaps_[idx])
        h = self.topo_.plot_map(axis, crange=crange)
        self.topo_.plot_locations(axis)
        self.topo_.plot_head(axis)
        return h
        
    def _plotUnmixing(self, axis, idx, crange=None):
        self.topo_.set_map(self.unmixmaps_[idx])
        h = self.topo_.plot_map(axis, crange=crange)
        self.topo_.plot_locations(axis)
        self.topo_.plot_head(axis)
        return h