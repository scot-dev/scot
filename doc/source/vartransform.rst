
Is VAR model fitting invariant to PCA transformations?
======================================================

:term:`MVARICA` usually applies a PCA transform to the EEG prior to VAR model fitting. This is intended as a
dimensionality reduction step; PCA components that contribute little to total EEG variance are removed. However, the PCA
produces orthogonal components. In other words, PCA transformed signals are uncorrelated.

The question was raised whether it is possible to reconstruct (fit) a VAR model from PCA transformed signals. Here we
show that this is, in fact, the case.

We will denote a var model with coefficients :math:`\mathbf{C}` and innovation process :math:`\vec{\epsilon}` as
:math:`\mathrm{VAR}(\mathbf{C},\vec{\epsilon})`.

Let's start with a VAR process :math:`\vec{x}_n = \mathrm{VAR}(\mathbf{A},\vec{e})`. If the model
contains causal structure, elements of :math:`\vec{x}` will in most cases show some degree of correlation. Let
:math:`\vec{y}_n = \mathbf{W} \vec{x}_n` be the PCA transformed signal. Furthermore, assume that :math:`\vec{y}` is a
VAR process too: :math:`\vec{y}_n = \mathrm{VAR}(\mathbf{B},\vec{r})`.

In order to reconstruct the original VAR model :math:`\mathrm{VAR}(\mathbf{A},\vec{e})` from
:math:`\mathrm{VAR}(\mathbf{B},\vec{r})` the following requirements need to be met:

  1. :math:`\mathrm{VAR}(\mathbf{B},\vec{r})` can be transformed into :math:`\mathrm{VAR}(\mathbf{A},\vec{e})` when the PCA transform :math:`\mathbf{W}` is known.
  2. A VAR model can have zero cross-correlation despite having causal structure.

The first requirement is obvious. Only when the models can be transformed into each other it is possible to reconstruct one model from another.
Since the PCA transformation :math:`\mathbf{W}` is a rotation matrix, its inverse is the transpose :math:`\mathbf{W}^{-1} = \mathbf{W}^\intercal`.
In section :ref:`lintransvar` we show that transformation of VAR models is possible if :math:`\mathbf{S} \mathbf{R} = \mathbf{I}` and :math:`\mathbf{R} \mathbf{S} = \mathbf{I}`.
This is the case with PCA since :math:`\mathbf{W}^\intercal \mathbf{W} = \mathbf{W} \mathbf{W}^\intercal = \mathbf{I}`.

The second requirement relates to the fact that in order to reconstruct model A from model B all information about A must be present in B.
Thus, information about the causal structure of A must be preserved in B, although :math:`\vec{y}_n = \mathrm{VAR}(\mathbf{B},\vec{r})` is uncorrelated.
:ref:`covbivar1` shows that it is possible to construct models where causal structure cancels cross-correlation.

In conclusion, it is possible to fit VAR models on PCA transformed signals and reconstruct the original model.


.. _lintransvar:

Linear transformation of a VAR model
------------------------------------
We start with a two VAR models; one for each vector signal :math:`\vec{x}` and :math:`\vec{y}`:

.. math::
    \vec{x}_n &= \sum_{k=1}^{p}\mathbf{A}^{(k)}\vec{x}_{n-k} + \vec{e}_n \\
    \vec{y}_n &= \sum_{k=1}^{p}\mathbf{B}^{(k)}\vec{y}_{n-k} + \vec{r}_n

Now assume that :math:`\vec{x}` and :math:`\vec{y}` can be transformed into each other by linear transformations:

.. math::
    \vec{y}_n &= \mathbf{R} \vec{x}_n \\
    \vec{x}_n &= \mathbf{S} \vec{y}_n

Note that

.. math::
    \left.
    \begin{matrix}
        \vec{y}_n = \mathbf{R} \mathbf{S} \vec{y}_n \\
        \vec{x}_n = \mathbf{S} \mathbf{R} \vec{x}_n
    \end{matrix}
    \right\} \Rightarrow \mathbf{R} \mathbf{S} = \mathbf{I}, \mathbf{S} \mathbf{R} = \mathbf{I}

By substituting the transformations into the VAR model equations we obtain

.. math::
    \vec{x}_n &= \sum_{k=1}^{p}\mathbf{S}\mathbf{B}^{(k)}\mathbf{R} \vec{x}_{n-k} + \mathbf{S} \vec{r}_n \\
    \vec{y}_n &= \sum_{k=1}^{p}\mathbf{R}\mathbf{A}^{(k)}\mathbf{S} \vec{y}_{n-k} + \mathbf{R} \vec{e}_n

Thus, each model can be transformed into the other by

.. math::
    \mathbf{A}^{(k)} &= \mathbf{S}\mathbf{B}^{(k)}\mathbf{R},\; \vec{e}_n = \mathbf{S} \vec{r}_n \\
    \mathbf{B}^{(k)} &= \mathbf{R}\mathbf{A}^{(k)}\mathbf{S},\; \vec{r}_n = \mathbf{R} \vec{e}_n

Conclusion: We can equivalently formulate VAR models for vector signals, if these signals are related by linear
transformations that satisfy :math:`\mathbf{S} \mathbf{R} = \mathbf{I}` and :math:`\mathbf{R} \mathbf{S} = \mathbf{I}`.


.. _covbivar1:

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
