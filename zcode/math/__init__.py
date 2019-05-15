"""Math and numerical routines.
"""

from . import math_core
from .math_core import *
from . import hist
from .hist import *
from . import numeric
from .numeric import *
from . import statistic
from .statistic import *
from . import kde
from .kde import *

__all__ = []
__all__.extend(math_core.__all__)
__all__.extend(hist.__all__)
__all__.extend(statistic.__all__)
__all__.extend(numeric.__all__)
__all__.extend(kde.__all__)

from numpy.testing import Tester
test = Tester().test
