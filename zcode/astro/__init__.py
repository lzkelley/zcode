"""Astrophysics submodule.
"""

from . import astro_core
from .astro_core import *  # noqa
from . import gws
from .gws import *  # noqa
from . import scalings
from .scalings import *  # noqa
from . import obs
from .obs import *  # noqa

__all__ = []
__all__.extend(astro_core.__all__)
__all__.extend(gws.__all__)
__all__.extend(scalings.__all__)
__all__.extend(obs.__all__)
