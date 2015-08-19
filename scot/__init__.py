# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

""" SCoT: The Source Connectivity Toolbox
"""

from __future__ import absolute_import

from . import config

# default backend
# TODO: set default backend in config
from . import backend_builtin

from .ooapi import Workspace

from .connectivity import Connectivity

from . import datatools

__all__ = ['Workspace', 'Connectivity', 'datatools']
__version__ = "0.1.0"

backends = ['backend_builtin', 'backend_sklearn']
