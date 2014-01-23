# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

""" Object oriented API to SCoT """

import numpy as np
from copy import deepcopy
from . import config
from .varica import mvarica
from .plainica import plainica
from .datatools import dot_special, randomize_phase
from .connectivity import Connectivity
from .connectivity_statistics import surrogate_connectivity, bootstrap_connectivity, test_bootstrap_difference, significance_fdr
from . import plotting
from eegtopo.topoplot import Topoplot


class Workspace:
    def __init__(self, var, locations=None, reducedim=0.99, nfft=512, fs=2, backend=None):
        """
        Workspace(var_order, **args)

        Create a new Workspace instance.

        Parameters     Default  Shape   Description
        --------------------------------------------------------------------------
        var            :      :       : Instance of a VAR model class, or dictionary
                                        with attributes to pass to the backend's
                                        default VAR constructor.

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
        self.connectivity_ = None
        self.locations_ = locations
        self.reducedim_ = reducedim
        self.nfft_ = nfft
        self.backend_ = backend

        self.trial_mask_ = []

        self.topo_ = None
        self.mixmaps_ = []
        self.unmixmaps_ = []

        self.var_multiclass_ = None

        self.plot_diagonal = 'topo'
        self.plot_outside_topo = False
        self.plot_f_range = [0, fs/2]

        if self.backend_ is None:
            self.backend_ = config.backend

        try:
            self.var_ = self.backend_['var'](**var)
        except TypeError:
            self.var_ = var


    def __str__(self):
        """Information about the Workspace."""

        if self.data_ is not None:
            datastr = '%d samples, %d channels, %d trials' % self.data_.shape
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

        if self.var_ is None:
            varstr = 'None'
        else:
            varstr = str(self.var_)

        s = 'Workspace:\n'
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
        self.cl_ = np.asarray(cl if cl is not None else [None]*self.data_.shape[2])
        self.time_offset_ = time_offset
        self.var_model_ = None
        self.var_cov_ = None
        self.connectivity_ = None

        self.trial_mask_ = np.ones(self.cl_.size, dtype=bool)

        if self.unmixing_ != None:
            self.activations_ = dot_special(self.data_, self.unmixing_)

    def set_used_labels(self, labels):
        mask = np.zeros((self.cl_.size), dtype=bool)
        for l in labels:
            mask = np.logical_or(mask, self.cl_ == l)
        self.trial_mask_ = mask

    def do_mvarica(self, varfit='ensemble'):
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
        if self.data_ is None:
            raise RuntimeError("MVARICA requires data to be set")
        result = mvarica(x=self.data_[:, :, self.trial_mask_], cl=self.cl_[self.trial_mask_], var=self.var_, reducedim=self.reducedim_, backend=self.backend_, varfit=varfit)
        self.mixing_ = result.mixing
        self.unmixing_ = result.unmixing
        self.var_ = result.b
        self.connectivity_ = Connectivity(result.b.coef, result.b.rescov, self.nfft_)
        self.activations_ = dot_special(self.data_, self.unmixing_)
        self.mixmaps_ = []
        self.unmixmaps_ = []
        return result

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
        if self.data_ is None:
            raise RuntimeError("ICA requires data to be set")
        result = plainica(x=self.data_[:,:,self.trial_mask_], reducedim=self.reducedim_, backend=self.backend_)
        self.mixing_ = result.mixing
        self.unmixing_ = result.unmixing
        self.activations_ = dot_special(self.data_, self.unmixing_)
        self.var_model_ = None
        self.var_cov_ = None
        self.connectivity_ = None
        self.mixmaps_ = []
        self.unmixmaps_ = []
        return result

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
        if self.unmixing_ is None or self.mixing_ is None:
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

        """
        if self.activations_ is None:
            raise RuntimeError("VAR fitting requires source activations (run do_mvarica first)")
        self.var_.fit(data=self.activations_[:, :, self.trial_mask_])
        self.connectivity_ = Connectivity(self.var_.coef, self.var_.rescov, self.nfft_)

    def optimize_var(self):
        """
        Workspace.optimize_regularization(skipstep=1)

        Optimize the var model's hyperparameters (such as regularization).

        Behaviour of this function is modified by the following attributes:
            var_
        """
        if self.activations_ is None:
            raise RuntimeError("VAR fitting requires source activations (run do_mvarica first)")

        self.var_.optimize(self.activations_[:, :, self.trial_mask_])

    def get_connectivity(self, measure, plot=False):
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

        cm = getattr(self.connectivity_, measure)()

        if plot is None or plot:
            fig = plot
            if self.plot_diagonal == 'fill':
                diagonal = 0
            elif self.plot_diagonal == 'S':
                diagonal = -1
                sm = np.abs(self.connectivity_.S())
                fig = plotting.plot_connectivity_spectrum(sm, fs=self.fs_, freq_range=self.plot_f_range,
                                                          diagonal=1, border=self.plot_outside_topo, fig=fig)
            else:
                diagonal = -1

            fig = plotting.plot_connectivity_spectrum(cm, fs=self.fs_, freq_range=self.plot_f_range,
                                                      diagonal=diagonal, border=self.plot_outside_topo, fig=fig)

            return cm, fig

        return cm

    def get_surrogate_connectivity(self, measures, repeats=100):
        """ Calculates surrogate connectivity for a multivariate time series by phase randomization.

            Parameters     Default  Shape   Description
            --------------------------------------------------------------------------
            measures       :      :       : String or list of strings. Each string is
                                            the (case sensitive) name of a connectivity
                                            measure to calculate. See documentation of
                                            Connectivity for supported measures.
                                            The function returns an ndarray if measures
                                            is a string, otherwise a dict is returned.
            repeats        : 100  : 1     : Number of surrogates to create.

            Output   Shape               Description
            --------------------------------------------------------------------------
            result : repeats, m,m,nfft : An ndarray of shape (repeats, m, m, nfft) is
                                         returned if measures is a string. If measures
                                         is a list of strings a dictionary is returned,
                                         where each key is the name of the measure, and
                                         the corresponding values are ndarrays of shape
                                         (repeats, m, m, nfft).
        """
        return surrogate_connectivity(measures, self.activations_[:, :, self.trial_mask_],
                                      self.var_, self.nfft_, repeats)

    def get_bootstrap_connectivity(self, measures, repeats=100, num_samples=None, plot=False):
        """ Calculates Bootstrap estimates of connectivity by randomly sampling trials with replacement.

            Parameters     Default  Shape   Description
            --------------------------------------------------------------------------
            measures       :      :       : String or list of strings. Each string is
                                            the (case sensitive) name of a connectivity
                                            measure to calculate. See documentation of
                                            Connectivity for supported measures.
                                            The function returns an ndarray if measures
                                            is a string, otherwise a dict is returned.
            num_samples    : None : 1     : Number of trials to sample for each estimate. Defaults: t
            repeats        : 100  : 1     : Number of bootstrap estimates to calculate

            Output   Shape               Description
            --------------------------------------------------------------------------
            result : repeats, m,m,nfft : An ndarray of shape (repeats, m, m, nfft) is
                                         returned if measures is a string. If measures
                                         is a list of strings a dictionary is returned,
                                         where each key is the name of the measure, and
                                         the corresponding values are ndarrays of shape
                                         (repeats, m, m, nfft).

        Requires: data set
        """
        if num_samples is None:
            num_samples = np.sum(self.trial_mask_)

        cb = bootstrap_connectivity(measures, self.activations_[:, :, self.trial_mask_],
                                    self.var_, self.nfft_, repeats, num_samples)

        if plot is None or plot:
            fig = plot
            if self.plot_diagonal == 'fill':
                diagonal = 0
            elif self.plot_diagonal == 'S':
                diagonal = -1
                sb = self.get_bootstrap_connectivity('absS', repeats, num_samples)
                sm = np.median(sb, axis=0)
                sl = np.percentile(sb, 2.5, axis=0)
                su = np.percentile(sb, 97.5, axis=0)
                fig = plotting.plot_connectivity_spectrum([sm, sl, su], fs=self.fs_, freq_range=self.plot_f_range,
                                                          diagonal=1, border=self.plot_outside_topo, fig=fig)
            else:
                diagonal = -1
            cm = np.median(cb, axis=0)
            cl = np.percentile(cb, 2.5, axis=0)
            cu = np.percentile(cb, 97.5, axis=0)
            fig = plotting.plot_connectivity_spectrum([cm, cl, cu], fs=self.fs_, freq_range=self.plot_f_range,
                                                      diagonal=diagonal, border=self.plot_outside_topo, fig=fig)
            return cb, fig

        return cb

    def get_tf_connectivity(self, measure, winlen, winstep, plot=False):
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
        if self.activations_ is None:
            raise RuntimeError("Time/Frequency Connectivity requires activations (call set_data after do_mvarica)")
        [n, m, _] = self.activations_.shape

        nstep = (n - winlen) // winstep

        result = np.zeros((m, m, self.nfft_, nstep), np.complex64)
        i = 0
        for j in range(0, n - winlen, winstep):
            win = np.arange(winlen) + j
            data = self.activations_[win, :, :]
            data = data[:, :, self.trial_mask_]
            self.var_.fit(data)
            con = Connectivity(self.var_.coef, self.var_.rescov, self.nfft_)
            result[:, :, :, i] = getattr(con, measure)()
            i += 1

        if plot is None or plot:
            fig = plot
            t0 = 0.5 * winlen / self.fs_ + self.time_offset_
            t1 = self.data_.shape[0] / self.fs_ - 0.5 * winlen / self.fs_ + self.time_offset_
            if self.plot_diagonal == 'fill':
                diagonal = 0
            elif self.plot_diagonal == 'S':
                diagonal = -1
                s = np.abs(self.get_tf_connectivity('S', winlen, winstep))
                fig = plotting.plot_connectivity_timespectrum(s, fs=self.fs_, crange=[np.min(s), np.max(s)],
                                                          freq_range=self.plot_f_range, time_range=[t0, t1],
                                                          diagonal=1, border=self.plot_outside_topo, fig=fig)
            else:
                diagonal = -1

            tfc = self._clean_measure(measure, result)
            if diagonal == -1:
                for m in range(tfc.shape[0]):
                    tfc[m, m, :, :] = 0
            fig = plotting.plot_connectivity_timespectrum(tfc, fs=self.fs_, crange=[np.min(tfc), np.max(tfc)],
                                                          freq_range=self.plot_f_range, time_range=[t0, t1],
                                                          diagonal=diagonal, border=self.plot_outside_topo, fig=fig)

            return result, fig

        return result

    def compare_conditions(self, labels1, labels2, measure, alpha=0.01, repeats=100, num_samples=None, plot=False):
        self.set_used_labels(labels1)
        ca = self.get_bootstrap_connectivity(measure, repeats, num_samples)
        self.set_used_labels(labels2)
        cb = self.get_bootstrap_connectivity(measure, repeats, num_samples)

        p = test_bootstrap_difference(ca, cb)
        s = significance_fdr(p, alpha)

        if plot is None or plot:
            fig = plot
            if self.plot_diagonal == 'topo':
                diagonal = -1
            elif self.plot_diagonal == 'fill':
                diagonal = 0
            elif self.plot_diagonal is 'S':
                diagonal = -1
                self.set_used_labels(labels1)
                sa = self.get_bootstrap_connectivity('absS', repeats, num_samples)
                sm = np.median(sa, axis=0)
                sl = np.percentile(sa, 2.5, axis=0)
                su = np.percentile(sa, 97.5, axis=0)
                fig = plotting.plot_connectivity_spectrum([sm, sl, su], fs=self.fs_, freq_range=self.plot_f_range,
                                                          diagonal=1, border=self.plot_outside_topo, fig=fig)

                self.set_used_labels(labels2)
                sb = self.get_bootstrap_connectivity('absS', repeats, num_samples)
                sm = np.median(sb, axis=0)
                sl = np.percentile(sb, 2.5, axis=0)
                su = np.percentile(sb, 97.5, axis=0)
                fig = plotting.plot_connectivity_spectrum([sm, sl, su], fs=self.fs_, freq_range=self.plot_f_range,
                                                          diagonal=1, border=self.plot_outside_topo, fig=fig)

                p_s = test_bootstrap_difference(ca, cb)
                s_s = significance_fdr(p_s, alpha)

                plotting.plot_connectivity_significance(s_s, fs=self.fs_, freq_range=self.plot_f_range,
                                                        diagonal=1, border=self.plot_outside_topo, fig=fig)
            else:
                diagonal = -1

            cm = np.median(ca, axis=0)
            cl = np.percentile(ca, 2.5, axis=0)
            cu = np.percentile(ca, 97.5, axis=0)

            fig = plotting.plot_connectivity_spectrum([cm, cl, cu], fs=self.fs_, freq_range=self.plot_f_range,
                                                      diagonal=diagonal, border=self.plot_outside_topo, fig=fig)

            cm = np.median(cb, axis=0)
            cl = np.percentile(cb, 2.5, axis=0)
            cu = np.percentile(cb, 97.5, axis=0)

            fig = plotting.plot_connectivity_spectrum([cm, cl, cu], fs=self.fs_, freq_range=self.plot_f_range,
                                                      diagonal=diagonal, border=self.plot_outside_topo, fig=fig)

            plotting.plot_connectivity_significance(s, fs=self.fs_, freq_range=self.plot_f_range,
                                                    diagonal=diagonal, border=self.plot_outside_topo, fig=fig)

            return p, s, fig

        return p, s

    @staticmethod
    def show_plots():
        """Show current plots."""
        plotting.show_plots()

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
        if self.unmixing_ is None and self.mixing_ is None:
            raise RuntimeError("No sources available (run do_mvarica first)")

        self._prepare_plots(True, True)

        plotting.plot_sources(self.topo_, self.mixmaps_, self.unmixmaps_, common_scale)

    def plot_connectivity_topos(self, fig=None):
        self._prepare_plots(True, False)
        if self.plot_outside_topo:
            fig = plotting.plot_connectivity_topos('outside', self.topo_, self.mixmaps_, fig)
        elif self.plot_diagonal == 'topo':
            fig = plotting.plot_connectivity_topos('diagonal', self.topo_, self.mixmaps_, fig)
        return fig

    def plot_connectivity_surrogate(self, measure, freq_range=(-np.inf, np.inf), repeats=100, fig=None):
        """
        Workspace.plot_connectivity_surrogate(measure, freq_range, repeats=100, fig=None)

        Plot spectral connectivity.

        Parameters     Default  Shape   Description
        --------------------------------------------------------------------------
        measure        :      : str   : Refer to scot.Connectivity for supported
                                        measures.
        freq_range     :      : 2     : Restrict plotted frequency range.
        repeats        : 100  : 1     : Number of surrogates to compute

        Requires: var model
        """
        cb = self.get_surrogate_connectivity(measure, repeats)

        self._prepare_plots(True, False)

        cu = np.percentile(cb, 95, axis=0)

        fig = plotting.plot_connectivity_spectrum([cu], self.fs_, freq_range=freq_range, fig=fig)

        return fig

    # def plot_tf_connectivity(self, measure, winlen, winstep, freq_range=(-np.inf, np.inf), crange=None, ignore_diagonal=True):
    #     """
    #     Workspace.plot_tf_connectivity(measure, winlen, winstep, freq_range)
    #
    #     Calculate and plot time-varying spectral connectivity measure.
    #
    #     Connectivity is estimated in a sliding window approach on the current
    #     data set.
    #
    #     Parameters     Default  Shape   Description
    #     --------------------------------------------------------------------------
    #     measure        :      : str   : Refer to scot.Connectivity for supported
    #                                     measures.
    #     winlen         :      :       : Length of the sliding window (in samples).
    #     winstep        :      :       : Step size for sliding window (in sapmles).
    #     freq_range     :      : 2     : Restrict plotted frequency range.
    #
    #     Requires: var model
    #     """
    #     t0 = 0.5 * winlen / self.fs_ + self.time_offset_
    #     t1 = self.data_.shape[0] / self.fs_ - 0.5 * winlen / self.fs_ + self.time_offset_
    #
    #     self._prepare_plots(True, False)
    #     tfc = self.get_tf_connectivity(measure, winlen, winstep)
    #
    #     if isinstance(tfc, dict):
    #         ncl = np.unique(self.cl_)
    #         lowest, highest = np.inf, -np.inf
    #         for c in ncl:
    #             tfc[c] = self._clean_measure(measure, tfc[c])
    #             if ignore_diagonal:
    #                 for m in range(tfc[c].shape[0]):
    #                     tfc[c][m, m, :, :] = 0
    #             highest = max(highest, np.max(tfc[c]))
    #             lowest = min(lowest, np.min(tfc[c]))
    #
    #         if crange is None:
    #             crange = [lowest, highest]
    #
    #         fig = {}
    #         for c in ncl:
    #             fig[c] = plotting.plot_connectivity_timespectrum(tfc[c], fs=self.fs_, crange=crange,
    #                                                              freq_range=freq_range, time_range=[t0, t1],
    #                                                              topo=self.topo_, topomaps=self.mixmaps_)
    #             fig[c].suptitle(str(c))
    #
    #     else:
    #         tfc = self._clean_measure(measure, tfc)
    #         if ignore_diagonal:
    #             for m in range(tfc.shape[0]):
    #                 tfc[m, m, :, :] = 0
    #         fig = plotting.plot_connectivity_timespectrum(tfc, fs=self.fs_, crange=[np.min(tfc), np.max(tfc)],
    #                                                       freq_range=freq_range, time_range=[t0, t1], topo=self.topo_,
    #                                                       topomaps=self.mixmaps_)
    #     return fig

    def _prepare_plots(self, mixing=False, unmixing=False):
        if self.locations_ is None:
            raise RuntimeError("Need sensor locations for plotting")

        if self.topo_ is None:
            self.topo_ = Topoplot()
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
