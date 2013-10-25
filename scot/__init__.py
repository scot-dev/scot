# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

""" SCoT: The Source Connectivity Toolbox """

__all__ = ['connectivity', 'datatools', 'matfiles', 'ooapi', 'plainica', 'plotting', 'var', 'varica', 'xvschema']

from . import config

# default backend
from .backend import builtin

from .ooapi import Workspace

from .connectivity import Connectivity

#from . import matfiles
#from . import datatools
#from . import xvschema

#from . import var

#from .varica import mvarica

#from . import plotting
