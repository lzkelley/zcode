"""General astrophysical functions.

Functions
---------

"""
from __future__ import absolute_import, division, print_function, unicode_literals
# from six.moves import xrange

import numpy as np
# import warnings
# import numbers

from zcode.constants import NWTG, SPLC, MPRT, SIGMA_T

__all__ = ['dynamical_time', 'chirp_mass', 'eddington_accretion', 'eddington_luminosity',
           'kepler_freq_from_sep', 'kepler_sep_from_freq', 'schwarzschild_radius']

_SCHW_CONST = 2*NWTG/np.square(SPLC)
_EDD_CONST = 4.0*np.pi*SPLC*NWTG*MPRT/SIGMA_T


def chirp_mass(m1, m2):
    return np.power(m1*m2, 3/5)/np.power(m1+m2, 1/5)


def dynamical_time(mass, rad):
    """Dynamical time of a gravitating system.
    """
    dt = 1/np.sqrt(NWTG*mass/rad**3)
    return dt


def eddington_accretion(mass, eps=0.1):
    """Eddington Accretion rate, $\dot{M}_{Edd} = L_{Edd}/\epsilon c^2$.^

    Arguments
    ---------
    mass : array_like of scalar
        BH Mass.
    eps : array_like of scalar
        Efficiency parameter.

    Returns
    -------
    mdot : array_like of scalar
        Eddington accretion rate.

    """
    edd_lum = eddington_luminosity(mass, eps=eps)
    mdot = edd_lum/(eps*np.square(SPLC))
    return mdot


def eddington_luminosity(mass, eps=0.1):
    ledd = _EDD_CONST * mass / eps
    return ledd


def kepler_freq_from_sep(mass, sep):
    freq = (1.0/(2.0*np.pi))*np.sqrt(NWTG*mass)/np.power(sep, 1.5)
    return freq


def kepler_sep_from_freq(mass, freq):
    sep = np.power(NWTG*mass/np.square(2.0*np.pi*freq), 1.0/3.0)
    return sep


def schwarzschild_radius(mass):
    rs = _SCHW_CONST * mass
    return rs
