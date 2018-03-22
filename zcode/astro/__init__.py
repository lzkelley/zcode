"""Astrophysics submodule.
"""

from . import astro_core
from .astro_core import *

__all__ = []
__all__.extend(astro_core.__all__)

from numpy.testing import Tester
test = Tester().test
