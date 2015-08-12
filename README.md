SCoT
====

SCoT is a Python package for EEG/MEG source connectivity estimation.


Obtaining SCoT
--------------
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


Examples
--------
To run the examples on Linux, invoke the following commands inside the SCoT main directory:

    PYTHONPATH=. python examples/misc/connectivity.py

    PYTHONPATH=. python examples/misc/timefrequency.py

etc.


Note that you need to obtain the example data from https://github.com/SCoT-dev/scot-data. The scot-data package must be on Python's search path.
