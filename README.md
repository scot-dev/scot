![Python 2.7](https://img.shields.io/badge/python-2.7-green.svg)
![Python 3.5](https://img.shields.io/badge/python-3.5-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
[![Build Status](https://travis-ci.org/scot-dev/scot.svg?branch=master)](https://travis-ci.org/scot-dev/scot)
[![Coverage Status](https://coveralls.io/repos/scot-dev/scot/badge.svg)](https://coveralls.io/r/scot-dev/scot)

SCoT
====

SCoT is a Python package for EEG/MEG source connectivity estimation.


Obtaining SCoT
--------------

##### From PyPi

Use the following command to install SCoT from PyPi:

    pip install scot


##### From Source

Use the following command to fetch the sources:

    git clone --recursive https://github.com/scot-dev/scot.git scot

The flag `--recursive` tells git to check out the numpydoc submodule, which is required for building the documentation.


Documentation
-------------
Documentation is available online at http://scot-dev.github.io/scot-doc/index.html.


Dependencies
------------
Required: numpy, scipy

Optional: matplotlib, scikit-learn

The lowest supported versions of these libraries are numpy 1.8.2, scipy 0.13.3, scikit-learn 0.15.0, and
matplotlib 1.4.0. Lower versions may work but are not tested.


Examples
--------
To run the examples on Linux, invoke the following commands inside the SCoT main directory:

    PYTHONPATH=. python examples/misc/connectivity.py

    PYTHONPATH=. python examples/misc/timefrequency.py

etc.


Note that you need to obtain the example data from https://github.com/SCoT-dev/scot-data. The scot-data package must be on Python's search path.

Note
----
As of version 0.2, the data format in all SCoT routines has changed. It is now consistent with Scipy and MNE-Python. Specifically, epoched input data is now arranged in three-dimensional arrays of shape `(epochs, channels, samples)`. In addition, continuous data is now arranged in two-dimensional arrays of shape `(channels, samples)`.