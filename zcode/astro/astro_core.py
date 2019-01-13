"""General astrophysical functions.

Functions
---------

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from zcode.constants import NWTG, SPLC, MPRT, SIGMA_T

__all__ = ['chirp_mass', 'distance', 'dynamical_time',
           'eddington_accretion', 'eddington_luminosity',
           'gw_hardening_rate_dadt', 'gw_strain_source_circ',
           'm1m2_from_mtmr', 'mtmr_from_m1m2',
           'kepler_freq_from_sep', 'kepler_sep_from_freq', 'rad_isco', 'schwarzschild_radius',
           'sep_to_merge_in_time', 'time_to_merge_at_sep']

_SCHW_CONST = 2*NWTG/np.square(SPLC)
_EDD_CONST = 4.0*np.pi*SPLC*NWTG*MPRT/SIGMA_T

# e.g. Sesana+2011 Eq.5
_GW_SRC_CONST = 8 * np.power(NWTG, 5/3) * np.power(2*np.pi, 2/3) / np.sqrt(10) / np.power(SPLC, 4)
_GW_HARD_CONST = - 64 * np.power(NWTG, 3) / 5 / np.power(SPLC, 5)


def chirp_mass(m1, m2):
    return np.power(m1*m2, 3/5)/np.power(m1+m2, 1/5)


def distance(x1, x0=None):
    if x0 is None:
        xx = x1
    else:
        xx = x1 - x0

    dist = np.sqrt(np.sum(np.square(xx), axis=-1))
    return dist


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


def gw_hardening_rate_dadt(m1, m2, sma, ecc=None):
    """GW Hardening rate (da/dt).

    See Peters 1964, Eq. 5.6
    """
    cc = _GW_HARD_CONST
    dadt = cc * m1 * m2 * (m1 + m2) / np.power(sma, 3)
    if ecc is not None:
        fe = _gw_hardening_ecc_func(ecc)
        dadt *= fe
    return dadt


def gw_strain_source_circ(mchirp, dist_lum, freq_orb_rest):
    """GW Strain from a single source in a circular orbit.
    """
    cc = _GW_SRC_CONST
    hs = cc * mchirp * np.power(mchirp*freq_orb_rest, 2/3) / dist_lum
    return hs


def kepler_freq_from_sep(mass, sep):
    freq = (1.0/(2.0*np.pi))*np.sqrt(NWTG*mass)/np.power(sep, 1.5)
    return freq


def kepler_sep_from_freq(mass, freq):
    sep = np.power(NWTG*mass/np.square(2.0*np.pi*freq), 1.0/3.0)
    return sep


def kepler_vel_from_freq(mass, freq):
    vel = np.power(NWTG*mass*(2.0*np.pi*freq), 1.0/3.0)
    return vel


def m1m2_from_mtmr(mt, mr):
    """Convert from total-mass and mass-ratio to individual masses.
    """
    m1 = mt/(1.0 + mr)
    m2 = mt - m1
    return m1, m2


def mtmr_from_m1m2(m1, m2=None):
    if m2 is not None:
        masses = np.stack([m1, m2], axis=-1)
    else:
        assert np.shape(m1)[-1] == 2, "If only `m1` is given, last dimension must be 2!"
        masses = np.asarray(m1)

    mtot = masses.sum(axis=-1)
    mrat = masses.min(axis=-1) / masses.max(axis=-1)
    return mtot, mrat


def schwarzschild_radius(mass):
    rs = _SCHW_CONST * mass
    return rs


def rad_isco(m1, m2, factor=3.0):
    """Inner-most Stable Circular Orbit, radius at which binaries 'merge'.
    """
    return factor * schwarzschild_radius(m1+m2)


def sep_to_merge_in_time(m1, m2, time):
    """The initial separation required to merge within the given time.

    See: [Peters 1964].
    """
    GW_CONST = 64*np.power(NWTG, 3.0)/(5.0*np.power(SPLC, 5.0))
    a1 = rad_isco(m1, m2)
    return np.power(GW_CONST*m1*m2*(m1+m2)*time - np.power(a1, 4.0), 1./4.)


def time_to_merge_at_sep(m1, m2, sep):
    """The time required to merge starting from the given initial separation.

    See: [Peters 1964].
    """
    GW_CONST = 64*np.power(NWTG, 3.0)/(5.0*np.power(SPLC, 5.0))
    a1 = rad_isco(m1, m2)
    delta_sep = np.power(sep, 4.0) - np.power(a1, 4.0)
    return delta_sep/(GW_CONST*m1*m2*(m1+m2))


def _gw_hardening_ecc_func(ecc):
    """GW Hardening rate eccentricitiy dependence F(e).

    See Peters 1964, Eq. 5.6
    """
    e2 = ecc*ecc
    num = 1 + (73/24)*e2 + (37/96)*e2*e2
    den = np.power(1 - e2, 7/2)
    fe = num / den
    return fe
