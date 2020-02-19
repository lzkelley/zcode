"""Math and numerical routines.
"""

from . import math_core
from .math_core import *
from . import hist
from .hist import *
from . import numeric
from .numeric import *
from . import interpolate
from .interpolate import *
from . import statistic
from .statistic import *

__all__ = []
__all__.extend(math_core.__all__)
__all__.extend(hist.__all__)
__all__.extend(numeric.__all__)
__all__.extend(interpolate.__all__)
__all__.extend(statistic.__all__)

from numpy.testing import Tester
test = Tester().test
