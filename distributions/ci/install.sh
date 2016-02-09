#!/bin/bash
# This script is meant to be called by the "install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variable defined
# in the .travis.yml in the top level folder of the project.

set -e

if [[ "$INSTALL_ATLAS" == "true" ]]; then
    sudo apt-get install -qq libatlas3gf-base libatlas-dev
fi

if [[ "$DISTRIB" == "conda" ]]; then
    # Deactivate the travis-provided virtual environment and setup a
    # conda-based environment instead
    deactivate

    # Use the miniconda installer for faster download / install of conda
    # itself
    wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -O miniconda.sh
    chmod +x miniconda.sh && ./miniconda.sh -b
    export PATH=/home/travis/miniconda3/bin:$PATH
    conda update --yes conda

    # Configure the conda environment and put it in the path using the
    # provided versions
    if [[ "$USE_MKL" == "true" ]]; then
        conda create -n testenv --yes python=$PYTHON_VERSION pip nose \
            numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION numpy scipy \
            scikit-learn=$SKLEARN_VERSION matplotlib=$MATPLOTLIB_VERSION \
            libgfortran
    else
        conda create -n testenv --yes python=$PYTHON_VERSION nomkl pip nose \
            numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION numpy scipy \
            scikit-learn=$SKLEARN_VERSION matplotlib=$MATPLOTLIB_VERSION \
            libgfortran
    fi
    source activate testenv

elif [[ "$DISTRIB" == "ubuntu" ]]; then
    # Use standard ubuntu packages in their default version
    sudo apt-get install -qq python-scipy python-nose python-pip python-matplotlib python-sklearn
fi

if [[ "$COVERAGE" == "true" ]]; then
    pip install coverage coveralls
fi

if [[ "$RUN_EXAMPLES" == "true" ]]; then
    git clone https://github.com/scot-dev/scot-data.git
    cd scot-data
    python setup.py install
    cd ..
fi
