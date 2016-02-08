#!/bin/bash
# This script is meant to be called by the "script" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

echo "Running tests..."

if [[ "$INSTALL_SCOT" == "true" ]]; then
    python setup.py install
    cd tests
fi

export LD_LIBRARY_PATH=/home/travis/miniconda3/envs/testenv/lib:$LD_LIBRARY_PATH

if [[ "$COVERAGE" == "true" ]]; then
    xvfb-run --server-args="-screen 0 1024x768x24" nosetests --with-coverage --cover-package=scot,eegtopo --cover-inclusive --cover-branches;
else
    xvfb-run --server-args="-screen 0 1024x768x24" nosetests;
fi

if [[ "$RUN_EXAMPLES" == "true" ]]; then
    if [[ "$INSTALL_SCOT" == "true" ]]; then
        xvfb-run --server-args="-screen 0 1024x768x24" find ../examples -type f -iname "*\.py" -exec python {} \;
    else
        PYTHONPATH=. xvfb-run --server-args="-screen 0 1024x768x24" find examples -type f -iname "*\.py" -exec python {} \;
    fi
fi

echo "Running tests... done."
