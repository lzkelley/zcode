"""

"""
# flake8: noqa  --- ignore imported but unused flake8 warnings

import os

from . import constants

_cwd = os.path.realpath(os.path.dirname(__file__))

fname_version = os.path.join(_cwd, 'VERSION')
# print(fname_version, os.path.exists(fname_version), os.path.realpath(fname_version))
with open(fname_version) as inn:
    version = inn.read().strip()

__author__ = "Luke Zoltan Kelley"
__version__ = version
__email__ = "lzkelley@gmail.com"
__status__ = "Development"

from . constants import *
from . import math as zmath
from . import inout as zio
from . import plot as zplot
from . import astro as zastro
