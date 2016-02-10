Overview
========

This is the documentation of SCoT, the EEG source connectivity toolbox in Python.
SCoT provides functionality for blind source decomposition and connectivity estimation.
Connectivity is estimated from spectral measures (such as :func:`~scot.connectivity.Connectivity.COH`, :func:`~scot.connectivity.Connectivity.PDC`, or :func:`~scot.connectivity.Connectivity.DTF`) using vector autoregressive (VAR) models.

Note that the documentation is work-in-progress. Most sections are still missing, but we will add them in the near future.
However, the :ref:`api_reference` is in a usable state.


Note
====
As of version 0.2, the data format in all SCoT routines has changed. It is now consistent with Scipy and MNE-Python. Specifically, epoched input data is now arranged in three-dimensional arrays of shape `(epochs, channels, samples)`. In addition, continuous data is now arranged in two-dimensional arrays of shape `(channels, samples)`.


Contents
========

.. toctree::
   :maxdepth: 2

   info.rst
   manual.rst
   api_reference.rst
   misc.rst
