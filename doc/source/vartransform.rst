
Is VAR model fitting invariant to linear transformations?
=========================================================

:term:`MVARICA` usually applies a PCA transform to the EEG prior to VAR model fitting. This is intended as a
dimensionality reduction step; PCA components that contribute little to total EEG variance are removed. However, the PCA
produces orthogonal components. In other words, PCA transformed signals are uncorrelated.


Covariance of a bivariate AR(1) process
---------------------------------------

Consider the bivariate AR(1) process given by

.. math::

    \left[ \begin{matrix} x_1(n) \\ x_2(n)  \end{matrix} \right] =
    \left[ \begin{matrix} a_{11} & a_{12} \\  a_{21} & a_{22} \end{matrix} \right]
    \left[ \begin{matrix} x_1(n-1) \\ x_2(n-1)  \end{matrix} \right] +
    \left[ \begin{matrix} c_{11} & c_{12} \\  c_{21} & c_{22} \end{matrix} \right]
    \left[ \begin{matrix} e_1(n) \\ e_2(n)  \end{matrix} \right]

where :math:`e_1` and :math:`e_2` are uncorrelated Gaussian white noise processes with zero mean and unit variance.

The process variances :math:`s_i^2` and covariance :math:`r` are obtained by solving the following system of equations [1]_:

.. math::

    \left( \mathbf{I} - \left[ \begin{matrix} a_{11}^2 & a_{12}^2 & 2a_{11}a_{12} \\
                                              a_{21}^2 & a_{22}^2 & 2a_{21}a_{22} \\
                                              a_{11}a_{21} & a_{12}a_{22} & a_{11}a_{22} + a_{12}a_{21}
                               \end{matrix} \right]
    \right)
    \left[ \begin{matrix} s_1^2 \\ s_2^2 \\ r \end{matrix} \right] = \left[ \begin{matrix} c_{11}^2 + c_{12}^2 \\ c_{21}^2 + c_{22}^2 \\ c_{11}c_{21} + c_{12}c_{22} \end{matrix} \right]

In general, a VAR model with causal structure (:math:`a_{12} \neq 0` and/or :math:`a_{21} \neq 0`) has some instantaneous correlation (non-zero covariance :math:`r \neq 0`) between signals.

Now, let's constrain the system to zero covariance :math:`r = 0`.

.. math::

    c_{11}c_{21} + c_{12}c_{22} + a_{11}a_{21}s_1^2 + a_{12}a_{22}s_2^2 = 0

    \left( \mathbf{I} - \left[ \begin{matrix} a_{11}^2 & a_{12}^2 \\
                                              a_{21}^2 & a_{22}^2
                               \end{matrix} \right]
    \right)
    \left[ \begin{matrix} s_1^2 \\ s_2^2 \end{matrix} \right] = \left[ \begin{matrix} c_{11}^2 + c_{12}^2 \\ c_{21}^2 + c_{22}^2 \end{matrix} \right]

Conclusion: it is possible to construct special cases where VAR processes with causal structure have no instantaneous correlation.

.. [1] http://books.google.at/books?id=_VHxE26QvXgC&pg=PA230&lpg=PA230&dq=cross-covariance+of+bivariate+AR%281%29+process&source=bl&ots=EiwYr1CA6x&sig=zMJwf8s1MXk5CTyf6CKw9JfKBDU&hl=en&sa=X&ei=cLnqUsqRO6ve7Aan84DYDQ&ved=0CDIQ6AEwAQ#v=onepage&q=cross-covariance%20of%20bivariate%20AR%281%29%20process&f=false