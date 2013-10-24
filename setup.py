#!/usr/bin/env python

from distutils.core import setup

setup(name='SCoT',
      version='0.1.0',
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

