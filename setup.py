#!/usr/bin/env python

from distutils.core import setup


versionfile = open('VERSION', 'r')
ver = versionfile.read().strip()
versionfile.close()


download_binica = True
if download_binica:
    from scot.backend_builtin.binica import binica
    try:
        binica([])
    except ValueError:
        pass


setup(name='SCoT',
      version=ver,
      description='Source Connectivity Toolbox',
      author='Martin Billinger',
      author_email='martin.billinger@tugraz.at',
      url='https://github.com/SCoT-dev/SCoT',
      packages=['scot', 
                'scot.backend',
                'scot.builtin',
                'scot.eegtopo',
                'scot.eegtopo.geometry'],

      package_data={'scot.builtin': ['binica/ica_linux']},

      install_requires=['numpy >=1.7', 'scipy >=0.12']
     )

