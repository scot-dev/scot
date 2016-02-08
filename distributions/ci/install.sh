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
    
    pushd .

    # Use the miniconda installer for faster download / install of conda
    # itself
    wget http://repo.continuum.io/miniconda/Miniconda-3.6.0-Linux-x86_64.sh \
        -O miniconda.sh
    chmod +x miniconda.sh && ./miniconda.sh -b
    cd ..
    export PATH=/home/travis/miniconda/bin:$PATH
    conda update --yes conda
    
    popd

    # Configure the conda environment and put it in the path using the
    # provided versions
    
    conda remove --yes --features mkl || echo "MKL feature removed"
    
    if [[ "$INSTALL_MKL" == "true" ]]; then
        conda create -n testenv --yes python=$PYTHON_VERSION pip nose \
            numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION numpy scipy \
            scikit-learn=$SKLEARN_VERSION matplotlib=$MATPLOTLIB_VERSION \
            libgfortran mkl
    else
        conda create -n testenv --yes python=$PYTHON_VERSION pip nose \
            numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION \
            scikit-learn=$SKLEARN_VERSION matplotlib=$MATPLOTLIB_VERSION \
            libgfortran
    fi
    source activate testenv

    #if [[ "$INSTALL_MKL" == "true" ]]; then
    #    # Make sure that MKL is used
    #    conda install --yes mkl mkl-rt
    #else
    #    # Make sure that MKL is not used
    #    conda remove --yes --features mkl || echo "MKL feature removed"
    #    conda remove --yes mkl mkl-rt || echo "MKL libraries removed"
    #fi

    #if [[ "$INSTALL_FORTRAN" == "true" ]]; then
    #    # Make sure that MKL is used
    #    conda install --yes libgfortran
    #fi
                        
    conda info
    
    #echo `ls /home/travis/miniconda3`
    #echo `ls /home/travis/miniconda3/envs/testenv/lib -lha`
    #echo `find /home/travis/miniconda3 | grep .so`

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

python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"
python -c "import sklearn; print('sklearn %s' % sklearn.__version__)"
python -c "import matplotlib; print('matplotlib %s' % matplotlib.__version__)"
