![Python](https://img.shields.io/pypi/pyversions/scot.svg?logo=python&logoColor=white)
[![PyPI](https://img.shields.io/pypi/v/scot)](https://pypi.org/project/scot/)
[![Docs](https://img.shields.io/badge/docs-passing-brightgreen)](https://scot-dev.github.io/scot-doc/index.html)
[![License](https://img.shields.io/github/license/scot-dev/scot)](LICENSE)

## SCoT

SCoT is a Python package for EEG/MEG source connectivity estimation. In particular, it includes measures of directed connectivity based on vector autoregressive modeling.


### Obtaining SCoT

Use the following command to install the latest release:

    pip install scot


### Documentation

Documentation is available at http://scot-dev.github.io/scot-doc/index.html.


### Dependencies

SCoT requires [numpy](http://www.numpy.org/) ≥ 1.8.2 and [scipy](https://scipy.org/) ≥ 0.13.3. Optionally, [matplotlib](https://matplotlib.org/) ≥ 1.4.0, [scikit-learn](https://scikit-learn.org/stable/) ≥ 0.15.0, and [mne](https://mne.tools/) ≥ 0.11.0 can be installed for additional functionality.


### Examples

To run the examples on Linux, invoke the following commands inside the SCoT directory:

    PYTHONPATH=. python examples/misc/connectivity.py

    PYTHONPATH=. python examples/misc/timefrequency.py

etc.

Note that the example data from https://github.com/SCoT-dev/scot-data needs to be available. The `scot-data` package must be on Python's search path.
