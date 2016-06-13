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
    conda create -n testenv --yes python=$PYTHON pip nose
    source activate testenv

    if [ -n $NUMPY ]; then
        conda install numpy=$NUMPY
    else
        conda install numpy
    fi

    if [ -n $SCIPY ]; then
        conda install scipy=$SCIPY
    else
        conda install scipy
    fi

    if [ -n $SKLEARN ]; then
        conda install scikit-learn=$SKLEARN
    else
        conda install scikit-learn
    fi

    if [ -n $MATPLOTLIB ]; then
        conda install matplotlib=$MATPLOTLIB
    else
        conda install numpy
    fi

    numpy scipy scikit-learn matplotlib
    pip install mne
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
