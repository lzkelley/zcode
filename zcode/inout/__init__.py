"""IO related methods.
"""

from . import inout_core
from inout_core import *
from . import log
from log import *

__all__ = []
__all__.extend(inout_core.__all__)
__all__.extend(log.__all__)