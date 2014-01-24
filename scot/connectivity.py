# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

""" Connectivity Analysis """

import numpy as np
from .utils import memoize


def connectivity(measure_names, b, c=None, nfft=512):
    """ calculate connectivity measures.

    Parameters
    ----------
    measure_names : {str, list of str}
        Name(s) of the connectivity measure(s) to calculate. See :class:`Connectivity` for supported measures.
    b : ndarray, shape = [n_channels, n_channels*model_order]
        VAR model coefficients. See :ref:`var-model-coefficients` for details about the arrangement of coefficients.
    c : ndarray, shape = [n_channels, n_channels], optional
        Covariance matrix of the driving noise process. Identity matrix is used if set to None.
    nfft : int, optional
        Number of frequency bins to calculate. Note that these points cover the range between 0 and half the
        sampling rate.

    Returns
    -------
    result : ndarray, shape = [n_channels, n_channels, `nfft`]
        An ndarray of shape [m, m, nfft] is returned if measures is a string. If measures is a list of strings a
        dictionary is returned, where each key is the name of the measure, and the corresponding values are ndarrays
        of shape [m, m, nfft].

    Notes
    -----
    It is more efficients to use this function to get more measures at once than calling the function multiple times.

    Examples
    --------
    >>> c = connectivity(['DTF', 'PDC'], [[0.3, 0.6], [0.0, 0.9]])
    """
    con = Connectivity(b, c, nfft)
    try:
        return getattr(con, measure_names)()
    except TypeError:
        return {m: getattr(con, m)() for m in measure_names}


#noinspection PyPep8Naming
class Connectivity:
    #TODO: Big optimization potential
    """ Calculation of connectivity measures
    
    This class calculates various spectral connectivity measures from a vector autoregressive (VAR) model.

    Parameters
    ----------
    b : ndarray, shape = [n_channels, n_channels*model_order]
        VAR model coefficients. See :ref:`var-model-coefficients` for details about the arrangement of coefficients.
    c : ndarray, shape = [n_channels, n_channels], optional
        Covariance matrix of the driving noise process. Identity matrix is used if set to None.
    nfft : int, optional
        Number of frequency bins to calculate. Note that these points cover the range between 0 and half the
        sampling rate.

    Methods
    -------
    :func:`A`
       Spectral representation of the VAR coefficients
    :func:`H`
        Transfer function that turns the innovation process into the VAR process
    :func:`S`
        Cross spectral density
    :func:`logS`
        Logarithm of the cross spectral density (S), for convenience.
    :func:`G`
        Inverse cross spectral density
    :func:`logG`
        Logarithm of the inverse cross spectral density
    :func:`PHI`
        Phase angle
    :func:`COH`
        Coherence
    :func:`pCOH`
        Partial coherence
    :func:`PDC`
        Partial directed coherence
    :func:`ffPDC`
        Full frequency partial directed coherence
    :func:`PDCF`
        PDC factor
    :func:`GPDC`
        Generalized partial directed coherence
    :func:`DTF`
        Directed transfer function
    :func:`ffDTF`
        Full frequency directed transfer function
    :func:`dDTF`
        Direct directed transfer function
    :func:`GDTF`
        Generalized directed transfer function

    Notes
    -----
    Connectivity measures are returned by member functions that take no arguments and return a matrix of
    shape [m,m,nfft]. The first dimension is the sink, the second dimension is the source, and the third dimension is
    the frequency.

    A summary of most supported measures can be found in [1]_.

    References
    ----------
    .. [1] Martin Billinger et al, “Single-trial connectivity estimation for classification of motor imagery data”,
           *J. Neural Eng.* 10, 2013.
    """

    def __init__(self, b, c=None, nfft=512):
        b = np.asarray(b)
        (m, mp) = b.shape
        p = mp // m
        if m * p != mp:
            raise AttributeError('Second dimension of b must be an integer multiple of the first dimension.')

        if c is None:
            self.c = None
        else:
            self.c = np.asarray(c)

        self.b = np.reshape(b, (m, m, p), 'c')
        self.m = m
        self.p = p
        self.nfft = nfft

    @memoize
    def Cinv(self):
        """ Inverse of the noise covariance
        """
        try:
            return np.linalg.inv(self.c)
        except np.linalg.linalg.LinAlgError:
            print('Warning: non invertible noise covariance matrix c!')
            return np.eye(self.c.shape[0])

    @memoize
    def A(self):
        """ Spectral VAR coefficients
        """
        return np.fft.rfft(np.dstack([np.eye(self.m), -self.b]), self.nfft * 2 - 1)

    @memoize
    def H(self):
        """ VAR transfer function
        """
        return _inv3(self.A())

    @memoize
    def S(self):
        """ Cross spectral density
        """
        if self.c is None:
            raise RuntimeError('Cross spectral density requires noise covariance matrix c.')
        H = self.H()
        return np.dstack([H[:, :, k].dot(self.c).dot(H[:, :, k].transpose().conj()) for k in range(self.nfft)])

    @memoize
    def logS(self):
        """ Logarithmic cross spectral density
        """
        return np.log10(np.abs(self.S()))

    @memoize
    def absS(self):
        """ Logarithmic cross spectral density
        """
        return np.abs(self.S())

    @memoize
    def G(self):
        """ Inverse cross spectral density
        """
        if self.c is None:
            raise RuntimeError('Inverse cross spectral density requires invertible noise covariance matrix c.')
        A = self.A()
        return np.dstack([A[:, :, k].transpose().conj().dot(self.Cinv()).dot(A[:, :, k]) for k in range(self.nfft)])

    @memoize
    def logG(self):
        """ Logarithmic inverse cross spectral density
        """
        return np.log10(np.abs(self.G()))

    @memoize
    def COH(self):
        """ Coherence
        """
        S = self.S()
        COH = np.zeros(S.shape, np.complex)
        for k in range(self.nfft):
            DS = S[:, :, k].diagonal()[np.newaxis]
            COH[:, :, k] = S[:, :, k] / np.sqrt(DS.transpose().dot(DS))
        return COH

    @memoize
    def PHI(self):
        """ Phase angle
        """
        return np.angle(self.S())

    @memoize
    def pCOH(self):
        """ Partial coherence
        """
        G = self.G()
        pCOH = np.zeros(G.shape, np.complex)
        for k in range(self.nfft):
            DG = G[:, :, k].diagonal()[np.newaxis]
            pCOH[:, :, k] = G[:, :, k] / np.sqrt(DG.transpose().dot(DG))
        return pCOH

    @memoize
    def PDC(self):
        """ Partial directed coherence
        """
        A = self.A()
        PDC = np.zeros(A.shape, np.complex)
        for k in range(self.nfft):
            for j in range(self.m):
                den = np.sqrt(A[:, j, k].transpose().conj().dot(A[:, j, k]))
                PDC[:, j, k] = A[:, j, k] / den
        return np.abs(PDC)

    @memoize
    def ffPDC(self):
        """ Full frequency partial directed coherence
        """
        A = self.A()
        PDC = np.zeros(A.shape, np.complex)
        for j in range(self.m):
            den = 0
            for k in range(self.nfft):
                den += A[:, j, k].transpose().conj().dot(A[:, j, k])
            PDC[:, j, :] = A[:, j, :] * self.nfft / np.sqrt(den)
        return np.abs(PDC)

    @memoize
    def PDCF(self):
        """ Partial directed coherence factor
        """
        A = self.A()
        PDCF = np.zeros(A.shape, np.complex)
        for k in range(self.nfft):
            for j in range(self.m):
                den = np.sqrt(A[:, j, k].transpose().conj().dot(self.Cinv()).dot(A[:, j, k]))
                PDCF[:, j, k] = A[:, j, k] / den
        return np.abs(PDCF)

    @memoize
    def GPDC(self):
        """ Generalized partial directed coherence
        """
        A = self.A()
        DC = np.diag(1 / np.diag(self.c))
        DS = np.sqrt(1 / np.diag(self.c))
        PDC = np.zeros(A.shape, np.complex)
        for k in range(self.nfft):
            for j in range(self.m):
                den = np.sqrt(A[:, j, k].transpose().conj().dot(DC).dot(A[:, j, k]))
                PDC[:, j, k] = A[:, j, k] * DS / den
        return np.abs(PDC)

    @memoize
    def DTF(self):
        """ Directed transfer function
        """
        H = self.H()
        DTF = np.zeros(H.shape, np.complex)
        for k in range(self.nfft):
            for i in range(self.m):
                den = np.sqrt(H[i, :, k].transpose().conj().dot(H[i, :, k]))
                DTF[i, :, k] = H[i, :, k] / den
        return np.abs(DTF)

    @memoize
    def ffDTF(self):
        """ Full frequency directed transfer function
        """
        H = self.H()
        DTF = np.zeros(H.shape, np.complex)
        for i in range(self.m):
            den = 0
            for k in range(self.nfft):
                den += H[i, :, k].transpose().conj().dot(H[i, :, k])
            DTF[i, :, :] = H[i, :, :] * self.nfft / np.sqrt(den)
        return np.abs(DTF)

    @memoize
    def dDTF(self):
        """" Direct" directed transfer function
        """
        return np.abs(self.pCOH()) * self.ffDTF()

    @memoize
    def GDTF(self):
        """ Generalized directed transfer function
        """
        H = self.H()
        DC = np.diag(np.diag(self.c))
        DS = np.sqrt(np.diag(self.c))
        DTF = np.zeros(H.shape, np.complex)
        for k in range(self.nfft):
            for i in range(self.m):
                den = np.sqrt(H[i, :, k].transpose().conj().dot(DC).dot(H[i, :, k]))
                DTF[i, :, k] = H[i, :, k] * DS / den
        return np.abs(DTF)


def _inv3(x):
    y = np.zeros(x.shape, np.complex)
    for k in range(x.shape[2]):
        y[:, :, k] = np.linalg.inv(x[:, :, k])
    return y
