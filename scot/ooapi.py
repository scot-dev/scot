# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

""" Object oriented API to SCoT """

import numpy as np
from .varica import mvarica
from .plainica import plainica
from .datatools import dot_special
from .connectivity import Connectivity
from . import plotting
from . import var
from eegtopo.topoplot import Topoplot


class Workspace:
    
    def __init__(self, var_order, var_delta=None, locations=None, reducedim=0.99, nfft=512, fs=2, backend=None):
        """
        Workspace(var_order, **args)

        Create a new Workspace instance.

        Parameters     Default  Shape   Description
        --------------------------------------------------------------------------
        var_order      :      :       : Autoregressive model order

        Opt. Parameters Default  Shape   Description
        --------------------------------------------------------------------------
        fs             : 2    :       : Sampling rate in 1/s. Defaults to 2 so 1
                                        corresponds to the Nyquist frequency.
        locations      : None : [Nx3] : Electrode locations in cartesian coordinates.
                                        (required for plotting)
        nfft           : 512  :       : Number of frequency bins for connectivity
                                        estimation.
        reducedim      : 0.99 :       : a number less than 1 is interpreted as the
                                        fraction of variance that should remain in
                                        the data. All components that describe in
                                        total less than 1-retain_variance of the
                                        variance in the data are removed by the PCA.
                                        An integer number of 1 or greater is
                                        interpreted as the number of components to
                                        keep after applying the PCA.
                                        If set to 'no_pca' the PCA step is skipped.
        var_delta      : None :       : Regularization parameter for VAR model fitting
        backend        : None :       : Specify backend to use. When set to None
                                        SCoT's default backend (see config.py)
                                        is used.
        """
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
        """Information about the Workspace."""
        
        if self.data_ is not None:
            datastr = '%d samples, %d channels, %d trials'%self.data_.shape
        else:
            datastr = 'None'
            
            
        if self.cl_ is not None:
            clstr = str(np.unique(self.cl_))
        else:
            clstr = 'None'

        if self.unmixing_ is not None:
            sourcestr = str(self.unmixing_.shape[1])
        else:
            sourcestr = 'None'
            
        if self.var_model_ is None:
            varstr = 'None'
        elif isinstance(self.var_model_, dict):
            varstr = str(len(self.var_model_))
        else:
            varstr = '1'
        
        s = 'SCoT(var_order = %d):\n'%self.var_order_
        s += '  Data      : ' + datastr + '\n'
        s += '  Classes   : ' + clstr + '\n'
        s += '  Sources   : ' + sourcestr + '\n'
        s += '  VAR models: ' + varstr + '\n'
        
        return s
    
    def set_data(self, data, cl=None, time_offset=0):
        """
        Workspace.set_data(data, cl=None, time_offset=0)

        Create a new Workspace instance.

        Parameters     Default  Shape   Description
        --------------------------------------------------------------------------
        data           :      : n,m,T : 3d data matrix (n samples, m signals, T trials)
                              : n,m   : 2d data matrix (n samples, m signals)
        cl             : None : T     : List of class labels associated with
                                        each trial. a class label can be any
                                        python type (string, number, ...) that
                                        can be used as a key in map.
        time_offset    : 0    :       : Time offset of the trials. Used for
                                        labelling the x-axis of time/frequency
                                        plots.

        Provides: data set, class labels

        Invalidates: var model
        """
        self.data_ = np.atleast_3d(data)
        self.cl_ = cl
        self.time_offset_ = time_offset
        self.var_model_ = None
        self.var_cov_ = None
        self.connectivity_ = None
        
        if self.unmixing_ != None:
            self.activations_ = dot_special(self.data_, self.unmixing_)
    
    def do_mvarica(self):
        """
        Workspace.do_mvarica()

        Perform MVARICA source decomposition and VAR model fitting.

        Requires: data set

        Provides: decomposition, activations, var model

        Behaviour of this function is modified by the following attributes:
            var_order_
            var_delta_
            reducedim_
            backend_

        """
        if self.data_ == None:
            raise RuntimeError("MVARICA requires data to be set")
        result = mvarica(x=self.data_, p=self.var_order_, reducedim=self.reducedim_, delta=self.var_delta_, backend=self.backend_)
        self.mixing_ = result.mixing
        self.unmixing_ = result.unmixing
        self.var_model_ = result.b
        self.var_cov_ = result.c
        self.var_delta_ = result.delta
        self.connectivity_ = Connectivity(self.var_model_, self.var_cov_, self.nfft_)
        self.activations_ = dot_special(self.data_, self.unmixing_)
        self.mixmaps_ = []
        self.unmixmaps_ = []
    
    def do_ica(self):
        """
        Workspace.do_ica()

        Perform plain ICA source decomposition.

        Requires: data set

        Provides: decomposition, activations

        Invalidates: var model

        Behaviour of this function is modified by the following attributes:
            reducedim_
            backend_

        """
        if self.data_ == None:
            raise RuntimeError("ICA requires data to be set")
        result = plainica(x=self.data_, reducedim=self.reducedim_, backend=self.backend_)
        self.mixing_ = result.mixing
        self.unmixing_ = result.unmixing
        self.activations_ = dot_special(self.data_, self.unmixing_)
        self.var_model_ = None
        self.var_cov_ = None
        self.connectivity_ = None
        self.mixmaps_ = []
        self.unmixmaps_ = []
        
    def remove_sources(self, sources):
        """
        Workspace.remove_sources(sources)

        Manually remove sources from the decomposition.

        Parameters     Default  Shape   Description
        --------------------------------------------------------------------------
        sources        :      :      : Indicate which components to remove
                                       (slice, int or array of ints)

        Requires: decomposition

        Invalidates: var model

        """
        if self.unmixing_ == None or self.mixing_ == None:
            raise RuntimeError("No sources available (run do_mvarica first)")
        self.mixing_ = np.delete(self.mixing_, sources, 0)
        self.unmixing_ = np.delete(self.unmixing_, sources, 1)
        if self.activations_ != None:
            self.activations_ = np.delete(self.activations_, sources, 1)
        self.var_model_ = None
        self.var_cov_ = None
        self.connectivity_ = None
        
    
    def fit_var(self):
        """
        Workspace.fit_var()

        Fit new VAR model(s).

        Requires: data set

        Provides: var model

        Behaviour of this function is modified by the following attributes:
            var_order_
            var_delta_
            cl_

        """
        if self.activations_ == None:
            raise RuntimeError("VAR fitting requires source activations (run do_mvarica first)")
        if self.cl_ is None:
            self.var_model_, self.var_cov_ = var.fit(data=self.activations_, p=self.var_order_, delta=self.var_delta_, return_covariance=True)
            self.connectivity_ = Connectivity(self.var_model_, self.var_cov_, self.nfft_)
        else:
            self.var_model_, self.var_cov_ = var.fit_multiclass(data=self.activations_, cl=self.cl_, p=self.var_order_, delta=self.var_delta_, return_covariance=True)
            self.connectivity_ = {}
            for c in np.unique(self.cl_):
                self.connectivity_[c] = Connectivity(self.var_model_[c], self.var_cov_[c], self.nfft_)

    def optimize_regularization(self, xvschema, skipstep=1):
        """
        Workspace.optimize_regularization(xvschema, skipstep=1)

        Attempt to find a close-to-optimal regularization Parameter for the
        current data set.

        Parameters     Default  Shape   Description
        --------------------------------------------------------------------------
        xvschema       :      : func  : Function to generate training and testing set.
                                        See scot.xvschema module.
        skipstep       : 1    : 1     : Higher values speed up the calculation but
                                        cause higher variance in cost function which
                                        will result in less accurate results.

        Requires: activations

        Behaviour of this function is modified by the following attributes:
            var_order_

        """
        if self.activations_ == None:
            raise RuntimeError("VAR fitting requires source activations (run do_mvarica first)")
            
        self.var_delta_ = var.optimize_delta_bisection(data=self.activations_, p=self.var_order_, xvschema=xvschema, skipstep=skipstep)
                
    
    def get_connectivity(self, measure):
        """
        Workspace.get_connectivity(measure)

        Calculate and return spectral connectivity measure.

        Parameters     Default  Shape   Description
        --------------------------------------------------------------------------
        measure        :      : str   : Refer to scot.Connectivity for supported
                                        measures.

        Requires: var model

        Behaviour of this function is modified by the following attributes:
            nfft_
            cl_
        """
        if self.connectivity_ is None:
            raise RuntimeError("Connectivity requires a VAR model (run do_mvarica or fit_var first)")
        if isinstance(self.connectivity_, dict):
            result = {}
            for c in np.unique(self.cl_):
                result[c] = getattr(self.connectivity_[c], measure)()
            return result
        else:
            return getattr(self.connectivity_, measure)()
    
    def get_tf_connectivity(self, measure, winlen, winstep):
        """
        Workspace.get_tf_connectivity(measure, winlen, winstep)

        Calculate and return time-varying spectral connectivity measure.

        Connectivity is estimated in a sliding window approach on the current
        data set.

        Parameters     Default  Shape   Description
        --------------------------------------------------------------------------
        measure        :      : str   : Refer to scot.Connectivity for supported
                                        measures.
        winlen         :      :       : Length of the sliding window (in samples).
        winstep        :      :       : Step size for sliding window (in sapmles).

        Requires: var model

        Behaviour of this function is modified by the following attributes:
            nfft_
            cl_
        """
        if self.activations_ == None:
            raise RuntimeError("Time/Frequency Connectivity requires activations (call set_data after do_mvarica)")
        [n,m,_] = self.activations_.shape
        
        nstep = (n-winlen)//winstep
        
        if self.cl_ is None:
            result = np.zeros((m, m, self.nfft_, nstep), np.complex64)
            i = 0
            for j in range(0, n-winlen, winstep):
                win = np.arange(winlen) + j
                data = self.activations_[win,:,:]                
                b, c = var.fit(data, p=self.var_order_, delta=self.var_delta_, return_covariance=True)
                con = Connectivity(b, c, self.nfft_)
                result[:,:,:,i] = getattr(con, measure)()
                i += 1
        
        else:
            result = {}
            for ci in np.unique(self.cl_):
                result[ci] = np.zeros((m, m, self.nfft_, nstep), np.complex128)
            i = 0
            for j in range(0, n-winlen, winstep):
                win = np.arange(winlen) + j
                data = self.activations_[win,:,:]                
                b, c = var.fit_multiclass(data, cl=self.cl_, p=self.var_order_, delta=self.var_delta_, return_covariance=True)
                for ci in result.keys():
                    con = Connectivity(b[ci], c[ci], self.nfft_)
                    result[ci][:,:,:,i] = getattr(con, measure)()
                i += 1
        return result
                
    @staticmethod
    def show_plots():
        """Show current plots."""
        plotting.show_plots( )
    
    def plot_source_topos(self, common_scale=None):
        """
        Workspace.plot_source_topos(common_scale=None)

        Plot topography of the Source decomposition.

        Parameters     Default  Shape   Description
        --------------------------------------------------------------------------
        common_scale   : None :       : If set to None, each topoplot's color
                                        axis is scaled individually.
                                        Otherwise specifies the percentile
                                        (1-99) of values in all plot. This value
                                        is taken as the maximum color scale.

        Requires: decomposition
        """
        if self.unmixing_ == None and self.mixing_ == None:
            raise RuntimeError("No sources available (run do_mvarica first)")
            
        self._prepare_plots(True, True)

        plotting.plot_sources(self.topo_, self.mixmaps_, self.unmixmaps_, common_scale)
    
    def plot_connectivity(self, measure, freq_range=(-np.inf, np.inf)):
        """
        Workspace.plot_connectivity(measure, freq_range)

        Plot spectral connectivity.

        Parameters     Default  Shape   Description
        --------------------------------------------------------------------------
        measure        :      : str   : Refer to scot.Connectivity for supported
                                        measures.
        freq_range     :      : 2     : Restrict plotted frequency range.

        Requires: var model
        """
        fig = None        
        self._prepare_plots(True, False)
        if isinstance(self.connectivity_, dict):            
            for c in np.unique(self.cl_):
                cm = getattr(self.connectivity_[c], measure)()
                fig = plotting.plot_connectivity_spectrum(cm, fs=self.fs_, freq_range=freq_range, topo=self.topo_, topomaps=self.mixmaps_, fig=fig)
        else:
            cm = getattr(self.connectivity_, measure)()
            fig = plotting.plot_connectivity_spectrum(cm, fs=self.fs_, freq_range=freq_range, topo=self.topo_, topomaps=self.mixmaps_)
        return fig
    
    def plot_tf_connectivity(self, measure, winlen, winstep, freq_range=(-np.inf, np.inf), ignore_diagonal=True):
        """
        Workspace.plot_tf_connectivity(measure, winlen, winstep, freq_range)

        Calculate and plot time-varying spectral connectivity measure.

        Connectivity is estimated in a sliding window approach on the current
        data set.

        Parameters     Default  Shape   Description
        --------------------------------------------------------------------------
        measure        :      : str   : Refer to scot.Connectivity for supported
                                        measures.
        winlen         :      :       : Length of the sliding window (in samples).
        winstep        :      :       : Step size for sliding window (in sapmles).
        freq_range     :      : 2     : Restrict plotted frequency range.

        Requires: var model
        """
        t0 = 0.5*winlen/self.fs_ + self.time_offset_
        t1 = self.data_.shape[0]/self.fs_ - 0.5*winlen/self.fs_ + self.time_offset_
        
        self._prepare_plots(True, False)
        tfc = self.get_tf_connectivity(measure, winlen, winstep)
        
        if isinstance(tfc, dict):
            ncl = np.unique(self.cl_).size
            lowest, highest = np.inf, -np.inf
            for c in ncl:
                tfc[c] = self._clean_measure(measure, tfc[c])
                if ignore_diagonal:
                    for m in range(tfc[c].shape[0]):
                        tfc[c][m,m,:,:] = 0
                highest = max(highest, np.max(tfc[c]))
                lowest = min(lowest, np.min(tfc[c]))
                
            fig = {}
            for c in ncl:
                fig[c] = plotting.plot_connectivity_timespectrum(tfc[c], fs=self.fs_, crange=[lowest, highest], freq_range=freq_range, time_range=[t0, t1], topo=self.topo_, topomaps=self.mixmaps_)
                
        else:
            tfc = self._clean_measure(measure, tfc)
            if ignore_diagonal:
                for m in range(tfc.shape[0]):
                    tfc[m,m,:,:] = 0
            fig = plotting.plot_connectivity_timespectrum(tfc, fs=self.fs_, crange=[np.min(tfc), np.max(tfc)], freq_range=freq_range, time_range=[t0, t1], topo=self.topo_, topomaps=self.mixmaps_)
        return fig
        
    def _prepare_plots(self, mixing=False, unmixing=False):
        if self.locations_ is None:
            raise RuntimeError("Need sensor locations for plotting")
            
        if self.topo_ is None:
            self.topo_ = Topoplot( )
            self.topo_.set_locations(self.locations_)
        
        if mixing and not self.mixmaps_:
            self.mixmaps_ = plotting.prepare_topoplots(self.topo_, self.mixing_)
        
        if unmixing and not self.unmixmaps_:
            self.unmixmaps_ = plotting.prepare_topoplots(self.topo_, self.unmixing_.transpose())
            
    @staticmethod
    def _clean_measure(measure, a):
        if measure in ['a', 'H', 'COH', 'pCOH']:
            return np.abs(a)
        elif measure in ['S', 'g']:
            return np.log(np.abs(a))
        else:
            return np.real(a)
