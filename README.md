![Python 3.5](https://img.shields.io/badge/python-3.5-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
[![Build Status](https://travis-ci.org/scot-dev/scot.svg?branch=master)](https://travis-ci.org/scot-dev/scot)
[![Coverage Status](https://codecov.io/gh/scot-dev/scot/branch/master/graphs/badge.svg)](https://codecov.io/gh/scot-dev/scot/branch/master)

## SCoT

SCoT is a Python package for EEG/MEG source connectivity estimation.


### Obtaining SCoT

Use the following command to install the latest release from PyPI:

    pip install scot


### Documentation

Documentation is available online at http://scot-dev.github.io/scot-doc/index.html.


### Dependencies

SCoT requires [numpy](http://www.numpy.org/) ≥ 1.8.2 and [scipy](https://scipy.org/) ≥ 0.13.3. Optionally, matplotlib ≥ 1.4.0, scikit-learn ≥ 0.15.0, and mne ≥ 0.11.0 can be installed for additional functionality.


### Examples

To run the examples on Linux, invoke the following commands inside the SCoT directory:

    PYTHONPATH=. python examples/misc/connectivity.py

    PYTHONPATH=. python examples/misc/timefrequency.py

etc.

Note that the example data from https://github.com/SCoT-dev/scot-data needs to be available. The `scot-data` package must be on Python's search path.
