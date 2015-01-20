#!/usr/bin/env python

from setuptools import setup
from codecs import open


with open('VERSION', encoding='utf-8') as version:
    ver = version.read().strip()

setup(
    name='scot',
    version=ver,
    description='Source Connectivity Toolbox',
    url='https://github.com/scot-dev/scot',
    author='SCoT Development Team',
    author_email='martin.billinger@tugraz.at',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
    ],
    keywords='source connectivity EEG MEG ICA',
    packages=['scot', 'scot.eegtopo', 'scot.external'],
    install_requires=['numpy >=1.7', 'scipy >=0.12'],
    )
