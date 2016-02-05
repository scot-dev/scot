# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013-2016 SCoT Development Team

import os

try:
    from configparser import ConfigParser
except ImportError:
    from ConfigParser import ConfigParser


def load_configuration():
    scotdir = os.path.abspath(os.path.dirname(__file__))
    config_files = [os.path.join(scotdir, 'scot.ini'),
                    '/etc/eegtopo.ini',
                    '/etc/scot.ini',
                    os.path.expanduser("~/.eegtopo.ini"),
                    os.path.expanduser("~/.scot.ini")]
    config = ConfigParser()
    files = config.read(config_files)
    if not files:
        raise ValueError('Could not parse configuration.')
    return config
