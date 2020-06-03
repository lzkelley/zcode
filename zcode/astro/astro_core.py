"""General astrophysical functions.

Functions
---------

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from zcode.constants import NWTG, SPLC, MPRT, SIGMA_T

__all__ = ['chirp_mass', 'distance', 'dynamical_time',
           'eddington_accretion', 'eddington_luminosity',
           'gw_hardening_rate_dadt', 'gw_dedt', 'dfdt_from_dadt',
           'gw_strain', 'gw_char_strain', 'gw_freq_dist_func',
           # 'gw_strain_source_circ',
           'm1m2_from_mtmr', 'mtmr_from_m1m2', 'orbital_velocities',
           'kepler_freq_from_sep', 'kepler_sep_from_freq', 'rad_isco', 'rad_isco_spin',
           'uniform_inclinations',
           'schwarzschild_radius', 'sep_to_merge_in_time', 'time_to_merge_at_sep',
           'rad_hill', 'rad_roche']

_SCHW_CONST = 2*NWTG/np.square(SPLC)
_EDD_CONST = 4.0*np.pi*SPLC*NWTG*MPRT/SIGMA_T

# e.g. Sesana+2004 Eq.36
#      http://adsabs.harvard.edu/abs/2004ApJ...611..623S
#      NOTE: THIS IS GW-FREQUENCY, NOT ORBITAL  [2020-05-29]
_GW_SRC_CONST = 8 * np.power(NWTG, 5/3) * np.power(np.pi, 2/3) / np.sqrt(10) / np.power(SPLC, 4)
_GW_DADT_SEP_CONST = - 64 * np.power(NWTG, 3) / 5 / np.power(SPLC, 5)
_GW_DEDT_ECC_CONST = - 304 * np.power(NWTG, 3) / 15 / np.power(SPLC, 5)


def chirp_mass(m1, m2=None):
    if m2 is None:
        m1, m2 = np.moveaxis(m1, -1, 0)
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
    # NOTE: no `epsilon` (efficiency) in this equation, because included in `eddington_luminosity`
    mdot = edd_lum/np.square(SPLC)
    return mdot


def eddington_luminosity(mass, eps=0.1):
    ledd = _EDD_CONST * mass / eps
    return ledd


def gw_hardening_rate_dadt(m1, m2, sma, ecc=None):
    """GW Hardening rate (da/dt).

    See Peters 1964, Eq. 5.6
    http://adsabs.harvard.edu/abs/1964PhRv..136.1224P
    """
    cc = _GW_DADT_SEP_CONST
    dadt = cc * m1 * m2 * (m1 + m2) / np.power(sma, 3)
    if ecc is not None:
        fe = _gw_hardening_ecc_func(ecc)
        dadt *= fe
    return dadt


def gw_dedt(m1, m2, sma, ecc):
    """GW Eccentricity Evolution rate (de/dt).

    See Peters 1964, Eq. 5.7
    http://adsabs.harvard.edu/abs/1964PhRv..136.1224P
    """
    cc = _GW_DEDT_ECC_CONST
    e2 = ecc**2
    dedt = cc * m1 * m2 * (m1 + m2) / np.power(sma, 4)
    dedt *= (1.0 + e2*121.0/304.0) * ecc / np.power(1 - e2, 5.0/2.0)
    return dedt


def gw_freq_dist_func(nn, ee=0.0):
    """Frequency Distribution Function.

    See [Enoki & Nagashima 2007](astro-ph/0609377) Eq. 2.4.
    This function gives g(n,e)

    FIX: use recursion relation when possible,
        J_{n-1}(x) + J_{n+1}(x) = (2n/x) J_n(x)
    """
    import scipy as sp
    import scipy.special  # noqa

    # Calculate with non-zero eccentrictiy
    bessel = sp.special.jn
    ne = nn*ee
    n2 = np.square(nn)
    jn_m2 = bessel(nn-2, ne)
    jn_m1 = bessel(nn-1, ne)
    jn = bessel(nn, ne)
    jn_p1 = bessel(nn+1, ne)
    jn_p2 = bessel(nn+2, ne)

    aa = np.square(jn_m2 - 2.0*ee*jn_m1 + (2/nn)*jn + 2*ee*jn_p1 - jn_p2)
    bb = (1 - ee*ee)*np.square(jn_m2 - 2*ee*jn + jn_p2)
    cc = (4.0/(3.0*n2)) * np.square(jn)
    gg = (n2*n2/32) * (aa + bb + cc)
    return gg


def dfdt_from_dadt(dadt, sma, mtot=None, freq_orb=None):
    if mtot is None and freq_orb is None:
        raise ValueError("Either `mtot` or `freq_orb` must be provided!")
    if freq_orb is None:
        freq_orb = kepler_freq_from_sep(mtot, sma)

    dfda = -(3.0/2.0) * (freq_orb / sma)
    dfdt = dfda * dadt
    return dfdt


def gw_char_strain(hs, dur_obs, freq_gw_obs, freq_gw_rst, dfdt):
    """

    See, e.g., Sesana+2004, Eq. 35
               http://adsabs.harvard.edu/abs/2004ApJ...611..623S

    Arguments
    ---------
    hs : array_like scalar
        Strain amplitude (e.g. `gw_strain()`, sky- and polarization- averaged)
    dur_obs : array_like scalar
        Duration of observations, in the observer frame


    """

    ncycles = freq_gw_rst**2 / dfdt
    ncycles = np.clip(ncycles, 0.0, dur_obs * freq_gw_obs)
    hc = hs * np.sqrt(ncycles)
    return hc


def gw_strain(mchirp, dlum, freq_gw_rest):
    """GW Strain from a single source in a circular orbit.

    e.g. Sesana+2004 Eq.36
         http://adsabs.harvard.edu/abs/2004ApJ...611..623S
         NOTE: THIS IS GW-FREQUENCY, NOT ORBITAL  [2020-05-29]

    """
    cc = _GW_SRC_CONST
    hs = cc * mchirp * np.power(mchirp*freq_gw_rest, 2/3) / dlum
    return hs


'''
def gw_strain_source_circ(mchirp, dist_lum, freq_orb_rest):
    """GW Strain from a single source in a circular orbit.
    """
    cc = _GW_SRC_CONST
    hs = cc * mchirp * np.power(mchirp*freq_orb_rest, 2/3) / dist_lum
    return hs
'''


def kepler_freq_from_sep(mass, sep):
    freq = (1.0/(2.0*np.pi))*np.sqrt(NWTG*mass)/np.power(sep, 1.5)
    return freq


def kepler_sep_from_freq(mass, freq):
    sep = np.power(NWTG*mass/np.square(2.0*np.pi*freq), 1.0/3.0)
    return sep


def kepler_vel_from_freq(mass, freq):
    vel = np.power(NWTG*mass*(2.0*np.pi*freq), 1.0/3.0)
    return vel


def orbital_velocities(mt, mr, per=None, sep=None):
    sep, per = _get_sep_per(mt, sep, per)
    v2 = np.power(NWTG*mt/sep, 1.0/2.0) / (1 + mr)
    # v2 = np.power(2*np.pi*NWTG*mt/per, 1.0/3.0) / (1 + mr)
    v1 = v2 * mr
    vels = np.moveaxis([v1, v2], 0, -1)
    return vels


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


def uniform_inclinations(shape):
    """Generate inclinations (0,pi) uniformly in sin(theta), i.e. spherically.
    """
    inclins = np.arccos(1 - np.random.uniform(0.0, 1.0, shape))
    return inclins


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


def rad_hill(sep, mrat):
    """Hill Radius / L1 Lagrangian Point

    See Eq.3.82 (and 3.75) in Murray & Dermott
    NOTE: this differs from other forms of the relation
    """
    rh = sep * np.power(mrat/3.0, 1.0/3.0)
    return rh


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
