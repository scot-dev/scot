# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013-2016 SCoT Development Team

"""Use scikit-learn routines as backend."""

from __future__ import absolute_import
import scipy as sp

from .datatools import atleast_3d, cat_trials, dot_special
from . import backend
from . import backend_builtin as builtin
from .varbase import VARBase


def generate():
    from sklearn.decomposition import FastICA
    from sklearn.decomposition import PCA

    def wrapper_fastica(data, random_state=None):
        """Call FastICA implementation from scikit-learn."""
        ica = FastICA(random_state=random_state)
        ica.fit(cat_trials(data).T)
        u = ica.components_.T
        m = ica.mixing_.T
        return m, u

    def wrapper_pca(x, reducedim):
        """Call PCA implementation from scikit-learn."""
        pca = PCA(n_components=reducedim)
        pca.fit(cat_trials(x).T)
        d = pca.components_
        c = pca.components_.T
        y = dot_special(c.T, x)
        return c, d, y

    class VAR(VARBase):
        """Scikit-learn based implementation of VAR class.

        This class fits VAR models using various implementations of generalized
        linear model fitting available in scikit-learn.

        Parameters
        ----------
        model_order : int
            Autoregressive model order.
        fitobj : class, optional
            Instance of a linear model implementation.
        n_jobs : int | None
            Number of jobs to run in parallel for various tasks (e.g. whiteness
            testing). If set to None, joblib is not used at all. Note that the
            main script must be guarded with `if __name__ == '__main__':` when
            using parallelization.
        verbose : bool
            Whether to print informations to stdout.
            Default: None - use verbosity from global configuration.
        """
        def __init__(self, model_order, fitobj=None, n_jobs=1, verbose=None):
            VARBase.__init__(self, model_order=model_order, n_jobs=n_jobs,
                             verbose=verbose)
            if fitobj is None:
                from sklearn.linear_model import LinearRegression
                fitobj = LinearRegression(fit_intercept=False)
            self.fitting_model = fitobj

        def fit(self, data):
            """Fit VAR model to data.

            Parameters
            ----------
            data : array, shape (trials, channels, samples)
                Continuous or segmented data set. If the data is continuous, a
                2D array of shape (channels, samples) can be provided.

            Returns
            -------
            self : :class:`VAR`
                The :class:`VAR` object.
            """
            data = atleast_3d(data)
            (x, y) = self._construct_eqns(data)
            self.fitting_model.fit(x, y)

            self.coef = self.fitting_model.coef_

            self.residuals = data - self.predict(data)
            self.rescov = sp.cov(cat_trials(self.residuals[:, :, self.p:]))

            return self

    backend = builtin.generate()
    backend.update({'ica': wrapper_fastica, 'pca': wrapper_pca, 'var': VAR})
    return backend


backend.register('sklearn', generate)
