Uploading SCoT to PyPI
======================

Prerequisites
-------------
Official documentation on how to upload to PyPI is available at https://packaging.python.org/en/latest/distributing.html. There is also an official sample project at https://github.com/pypa/sampleproject.

Required Python packages:

* pip
* setuptools
* wheel
* twine

Required files (in project root):

* setup.py
* setup.cfg (optional)
* README.rst or README.md

Build wheel
-----------
Since SCoT is pure Python and runs under 2 and 3, we can build a universal wheel as follows:

    python setup.py bdist_wheel --universal

The `--universal` flag can also be omitted if this option is set in `setup.cfg`:

    [bdist_wheel]
    universal=1

Register at PyPI
----------------
This step is only necessary if the project has not been registered before. It is necessary to register a user account there as well (note that PyPI and TestPyPI use different users databases, so it is necessary to register at both sites separately).

It is also recommended to register the project via the webform.

Upload to PyPI
--------------
To upload the wheel, simply execute:

    twine upload dist/*

To upload to a specific site (e.g. TestPyPI), type:

    twine upload -r testpypi dist/*

This command reads its configuration from `~/.pypirc`, which might look as follows:

    [distutils]
    index-servers=
        pypi
        testpypi

    [testpypi]
    repository = https://testpypi.python.org/pypi
    username =
    password =

    [pypi]
    repository = https://pypi.python.org/pypi
    username =
    password =
