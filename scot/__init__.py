# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013-2015 SCoT Development Team

""" SCoT: The Source Connectivity Toolbox
"""

from __future__ import absolute_import

from . import config
config = config.load_configuration()

from .backendmanager import BackendManager
backend = BackendManager()

# register backends that come shipped with scot
from . import backend_builtin
from . import backend_sklearn
backend.activate(config.get('scot', 'backend'))

from .ooapi import Workspace

from .connectivity import Connectivity

from . import datatools

__all__ = ['Workspace', 'Connectivity', 'datatools']

__version__ = "0.2.dev0"
