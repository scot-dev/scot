#!/usr/bin/env python

from distutils.core import setup


versionfile = open('VERSION', 'r')
ver = versionfile.read().strip()
versionfile.close()


download_binica = True
if download_binica:
    from scot.binica import binica
    try:
        binica([])
    except ValueError:
        pass


setup(name='SCoT',
      version=ver,
      description='Source Connectivity Toolbox',
      author='Martin Billinger',
      author_email='martin.billinger@tugraz.at',
      url='https://github.com/scot-dev/scot',
      packages=['scot',
                'scot.eegtopo'],

      package_data={'scot': ['binica/ica_linux']},

      install_requires=['numpy >=1.7', 'scipy >=0.12']
     )

