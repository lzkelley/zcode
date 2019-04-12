"""Plotting methods.
"""
# flake8: noqa  --- ignore imported but unused flake8 warnings

LW_CONF = 1.0
LW_OUTLINE = 0.6

COL_CORR = 'royalblue'

_PAD = 0.01

from . import plot_core
from .plot_core import *
from . import Hist2D
from .Hist2D import *
from . import CorrelationGrid
from .CorrelationGrid import *
from . import color2d
from .color2d import *
from . import layout
from .layout import *
from . import draw
from .draw import *

__all__ = []
__all__.extend(plot_core.__all__)
__all__.extend(Hist2D.__all__)
__all__.extend(CorrelationGrid.__all__)
__all__.extend(color2d.__all__)
__all__.extend(layout.__all__)
__all__.extend(draw.__all__)
