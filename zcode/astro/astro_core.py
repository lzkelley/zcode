"""General astrophysical functions.

Functions
---------

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from zcode.constants import NWTG, SPLC, MPRT, SIGMA_T
from zcode import utils

__all__ = [
    'dfdt_from_dadt', 'distance', 'dynamical_time',
    'eddington_accretion', 'eddington_luminosity',
    'kepler_freq_from_sep', 'kepler_sep_from_freq',
    'mtmr_from_m1m2', 'm1m2_from_mtmr', 'orbital_velocities',
    'rad_hill', 'rad_isco', 'rad_isco_spin', 'rad_roche',
    'uniform_inclinations', 'schwarzschild_radius',
]

_SCHW_CONST = 2*NWTG/np.square(SPLC)
_EDD_CONST = 4.0*np.pi*SPLC*NWTG*MPRT/SIGMA_T

# e.g. Sesana+2004 Eq.36
#      http://adsabs.harvard.edu/abs/2004ApJ...611..623S
#      NOTE: THIS IS GW-FREQUENCY, NOT ORBITAL  [2020-05-29]
# _GW_SRC_CONST = 8 * np.power(NWTG, 5/3) * np.power(np.pi, 2/3) / np.sqrt(10) / np.power(SPLC, 4)
# _GW_DADT_SEP_CONST = - 64 * np.power(NWTG, 3) / 5 / np.power(SPLC, 5)
# _GW_DEDT_ECC_CONST = - 304 * np.power(NWTG, 3) / 15 / np.power(SPLC, 5)


def dfdt_from_dadt(dadt, sma, mtot=None, freq_orb=None):
    if mtot is None and freq_orb is None:
        raise ValueError("Either `mtot` or `freq_orb` must be provided!")
    if freq_orb is None:
        freq_orb = kepler_freq_from_sep(mtot, sma)

    dfda = -(3.0/2.0) * (freq_orb / sma)
    dfdt = dfda * dadt
    return dfdt


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
    # NOTE: no `epsilon` (efficiency) in this equation, because included in `eddington_luminosity`
    mdot = edd_lum/np.square(SPLC)
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


def kepler_vel_from_freq(mass, freq):
    vel = np.power(NWTG*mass*(2.0*np.pi*freq), 1.0/3.0)
    return vel


def mtmr_from_m1m2(m1, m2=None):
    if m2 is not None:
        masses = np.stack([m1, m2], axis=-1)
    else:
        assert np.shape(m1)[-1] == 2, "If only `m1` is given, last dimension must be 2!"
        masses = np.asarray(m1)

    mtot = masses.sum(axis=-1)
    mrat = masses.min(axis=-1) / masses.max(axis=-1)
    return np.array([mtot, mrat])


def m1m2_from_mtmr(mt, mr):
    """Convert from total-mass and mass-ratio to individual masses.
    """
    mt = np.asarray(mt)
    mr = np.asarray(mr)
    m1 = mt/(1.0 + mr)
    m2 = mt - m1
    return np.array([m1, m2])


def orbital_velocities(mt, mr, per=None, sep=None):
    sep, per = _get_sep_per(mt, sep, per)
    v2 = np.power(NWTG*mt/sep, 1.0/2.0) / (1 + mr)
    # v2 = np.power(2*np.pi*NWTG*mt/per, 1.0/3.0) / (1 + mr)
    v1 = v2 * mr
    vels = np.moveaxis([v1, v2], 0, -1)
    return vels


def rad_hill(sep, mrat):
    """Hill Radius / L1 Lagrangian Point

    See Eq.3.82 (and 3.75) in Murray & Dermott
    NOTE: this differs from other forms of the relation
    """
    rh = sep * np.power(mrat/3.0, 1.0/3.0)
    return rh


def rad_isco(m1, m2, factor=3.0):
    """Inner-most Stable Circular Orbit, radius at which binaries 'merge'.
    """
    return factor * schwarzschild_radius(m1+m2)


def rad_isco_spin(mass, spin=0.0):
    """Inner-most stable circular orbit for a spinning BH.

    See:: Eq. 17-19 of Middleton-2015 - 1507.06153

    Arguments
    ---------
    mm : arraylike scalar,
        Mass of the blackhole in grams
    aa : arraylike scalar,
        Dimensionless spin of the blackhole (`a = Jc/GM^2`)
        NOTE: this should be positive for co-rotating, and negative for counter-rotating.

    Returns
    -------
    risco : arraylike scalar
        Radius of the ISCO, in centimeters

    """
    risco = schwarzschild_radius(mass) / 2.0
    if np.all(spin == 0.0):
        return 6*risco

    a2 = spin**2
    z1 = 1 + np.power(1 - a2, 1/3) * ((1 + spin)**(1/3) + (1 - spin)**(1/3))
    z2 = np.sqrt(3*a2 + z1**2)
    risco *= (3 + z2 + -1 * np.sign(spin) * np.sqrt((3-z1)*(3+z1+2*z2)))
    return risco


def rad_roche(sep, mfrac, ecc=0.0):
    """Average Roche-Lobe radius from [Eggleton-1983]/[Miranda+Lai-2015]

    NOTE: [Miranda+Lai-2015] give this form of the equation, with eccentricity dependence, and
          cite [Eggleton-1983], however, that paper includes no eccentricity dependence.
          The eccentricity dependence comes from switching from semi-major-axis to pericenter,
          which is appropriate based on model-dependent assumptions, but not universally.

    Arguments
    ---------
    sep : binary separation / semi-major-axis
    mfrac : mass fraction of the target object, defined as M_i/(M1+M2)
    ecc : binary eccentricity

    """
    # This is M_2 / M_1 if mfrac is $M_2 / (M1+M2)$, and $M_1/M_2$ if it is $M1 / (M1+M2)$.
    qq = mfrac / (1 - mfrac)

    # Note that these papers use `q \equiv M1/M2` which is `1/mrat` as we define it here
    q13 = np.power(qq, 1.0/3.0)
    q23 = q13**2
    rl = 0.49 * q23 * sep / (0.6*q23 + np.log(1.0 + q13))
    if ecc != 0.0:
        rl *= (1.0 - ecc)

    return rl


def schwarzschild_radius(mass):
    rs = _SCHW_CONST * mass
    return rs


def inclinations_uniform(shape):
    """Generate inclinations (0,pi) uniformly in sin(theta), i.e. spherically.
    """
    inclins = np.arccos(1 - np.random.uniform(0.0, 1.0, shape))
    return inclins


def uniform_inclinations(*args, **kwargs):
    utils.dep_warn("astro_core.uniform_inclinations", newname="astro_core.inclinations_uniform")
    return inclinations_uniform(*args, **kwargs)


def _get_sep_per(mt, sep, per):
    if (per is None) and (sep is None):
        raise ValueError("Either `per` or `sep` must be provided!")
    if (per is not None) and (sep is not None):
        raise ValueError("Only one of `per` or `sep` should be provided!")

    if per is None:
        per = 1 / kepler_freq_from_sep(mt, sep)

    if sep is None:
        sep = kepler_sep_from_freq(mt, 1/per)

    return sep, per
