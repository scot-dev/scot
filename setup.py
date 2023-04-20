#!/usr/bin/env python

from setuptools import setup
from codecs import open
from scot import __version__ as ver


with open('README.md', encoding='utf-8') as readme:
    long_description = readme.read()

setup(
    name='scot',
    version=ver,
    description='EEG/MEG Source Connectivity Toolbox',
    long_description=long_description,
    url='https://github.com/scot-dev/scot',
    author='SCoT Development Team',
    author_email='scotdev@googlegroups.com',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5'
    ],
    keywords='source connectivity EEG MEG ICA',
    packages=['scot', 'scot.eegtopo', 'scot.external'],
    package_data={'scot': ['scot.ini']}
)
