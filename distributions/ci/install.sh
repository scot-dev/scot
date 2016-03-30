#!/bin/bash

# This script is meant to be called by the "install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variables defined
# in the .travis.yml in the top level folder of the project.

set -e

if [[ "$DISTRIB" == "conda" ]]; then
    # deactivate the travis-provided virtual environment and setup a
    # conda-based environment instead
    deactivate
    wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -O miniconda.sh
    chmod +x miniconda.sh && ./miniconda.sh -b
    export PATH=/home/travis/miniconda3/bin:$PATH
    conda update --yes conda

    # configure and activate the conda environment
    if [[ "$USE_MKL" == "true" ]]; then
        conda create -n testenv --yes python=$PYTHON_VERSION pip nose \
            numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION numpy scipy \
            scikit-learn=$SKLEARN_VERSION matplotlib=$MATPLOTLIB_VERSION
    else
        conda create -n testenv --yes python=$PYTHON_VERSION nomkl pip nose \
            numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION numpy scipy \
            scikit-learn=$SKLEARN_VERSION matplotlib=$MATPLOTLIB_VERSION
    fi
    source activate testenv
    pip install mne=$MNE_VERSION
fi

if [[ "$COVERAGE" == "true" ]]; then
    pip install codecov
fi

if [[ "$RUN_EXAMPLES" == "true" ]]; then
    git clone https://github.com/scot-dev/scot-data.git
    cd scot-data
    python setup.py install
    cd ..
fi
