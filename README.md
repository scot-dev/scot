SCoT
====

EEG Source Connectivity Toolbox in Python


Obtaining SCoT
--------------

Use the following command to fetch the sources:

    git clone --recursive https://github.com/SCoT-dev/SCoT.git SCoT
    
`--recursive` tells git to check out the numpydoc submodule which is required for building the documentation.


Documentation
-------------
Documentation is available online at http://scot-dev.github.io/scot-doc/index.html


Dependencies
------------
Required: numpy, scipy

Optional: matplotlib, scikit-learn


Examples
--------

To run the examples on Linux invoke the following commands inside the SCoT main directory:

PYTHONPATH=. python examples/example_01_connectivity.py

PYTHONPATH=. python examples/example_02_timefrequency.py

etc.


Note that as of March 3 2014 you need to get the example data from https://github.com/SCoT-dev/scot-data. The scotdata package must be on Python's search path.
