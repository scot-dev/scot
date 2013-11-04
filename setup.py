#!/usr/bin/env python2

from distutils.core import setup


versionfile = open('VERSION', 'r')
ver = versionfile.read().strip()
versionfile.close()


setup(name='SCoT',
      version=ver,
      description='Source Connectivity Toolbox',
      author='Martin Billinger',
      author_email='martin.billinger@tugraz.at',
      url='https://github.com/SCoT-dev/SCoT',
      packages=['scot', 
                'scot.backend',
                'scot.builtin',
                'eegtopo',
                'eegtopo.geometry'],

      install_requires=['numpy >=1.7', 'scipy >=0.12']
     )

