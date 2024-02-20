![Python](https://img.shields.io/pypi/pyversions/scot.svg?logo=python&logoColor=white)
[![PyPI](https://img.shields.io/pypi/v/scot)](https://pypi.org/project/scot/)
[![Docs](https://img.shields.io/badge/docs-passing-brightgreen)](https://scot-dev.github.io/scot-doc/index.html)
[![DOI](https://img.shields.io/badge/doi-10.3389%2Ffninf.2014.00022-4c1)](https://doi.org/10.3389/fninf.2014.00022)
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


### Building the docs

In February 2024 we managed to build the docs with the following package versions:

```
[tool.poetry.dependencies]
python = "^3.11"
sphinx = "^7.2.6"
matplotlib = "^3.8.3"
scipy = "^1.12.0"
scikit-learn = "^1.4.1.post1"
```

Note that these are the most recent versions at the moment, so it is likely that future versions will just work.
When using a newer version of sphinx, it may be necessary to update the subrepository in doc/sphinxext/numpydoc.

