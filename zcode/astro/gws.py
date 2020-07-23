"""Gravitational Waves related functions

References:
- EN07 : [Enoki & Nagashima 2007](https://ui.adsabs.harvard.edu/abs/2007PThPh.117..241E/abstract)
- Sesana+2004 : [Sesana+2004](http://adsabs.harvard.edu/abs/2004ApJ...611..623S)
"""


import numpy as np

from zcode.constants import NWTG, SPLC

__all__ = [
    'chirp_mass',
    'gw_hardening_rate_dadt', 'gw_dedt',
    # 'gw_strain',
    'gw_strain_source',
    'gw_char_strain', 'gw_freq_dist_func', '_gw_freq_dist_func_old', 'gw_lum_circ',
    # 'gw_strain_source_circ',
    'sep_to_merge_in_time', 'time_to_merge_at_sep',
]

_SCHW_CONST = 2*NWTG/np.square(SPLC)

# e.g. Sesana+2004 Eq.36
_GW_SRC_CONST = 8 * np.power(NWTG, 5/3) * np.power(np.pi, 2/3) / np.sqrt(10) / np.power(SPLC, 4)
_GW_DADT_SEP_CONST = - 64 * np.power(NWTG, 3) / 5 / np.power(SPLC, 5)
_GW_DEDT_ECC_CONST = - 304 * np.power(NWTG, 3) / 15 / np.power(SPLC, 5)

# EN07, Eq.2.2
_LUM_CONST = (32.0 / 5.0) * np.power(NWTG, 7.0/3.0) * np.power(SPLC, -5.0)

from . import astro_core


def chirp_mass(m1, m2=None):
    if m2 is None:
        m1, m2 = np.moveaxis(m1, -1, 0)
    return np.power(m1*m2, 3/5)/np.power(m1+m2, 1/5)


def gw_char_strain(hs, dur_obs, freq_orb_obs, freq_orb_rst, dfdt):
    """

    See, e.g., Sesana+2004, Eq.35

    Arguments
    ---------
    hs : array_like scalar
        Strain amplitude (e.g. `gw_strain()`, sky- and polarization- averaged)
    dur_obs : array_like scalar
        Duration of observations, in the observer frame


    """

    ncycles = freq_orb_rst**2 / dfdt
    ncycles = np.clip(ncycles, None, dur_obs * freq_orb_obs)
    hc = hs * np.sqrt(ncycles)
    return hc


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

    # jn = bessel(nn, ne)
    # jn_p1 = bessel(nn+1, ne)
    # jn_p2 = bessel(nn+2, ne)

    # Use recursion relation:
    jn = (2*(nn-1) / ne) * jn_m1 - jn_m2
    jn_p1 = (2*nn / ne) * jn - jn_m1
    jn_p2 = (2*(nn+1) / ne) * jn_p1 - jn

    aa = np.square(jn_m2 - 2.0*ee*jn_m1 + (2/nn)*jn + 2*ee*jn_p1 - jn_p2)
    bb = (1 - ee*ee)*np.square(jn_m2 - 2*ee*jn + jn_p2)
    cc = (4.0/(3.0*n2)) * np.square(jn)
    gg = (n2*n2/32) * (aa + bb + cc)
    return gg


def _gw_freq_dist_func_old(nn, ee=0.0):
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


def gw_hardening_rate_dadt(m1, m2, sma, ecc=None):
    """GW Hardening rate (da/dt).

    See Peters 1964, Eq. 5.6
    http://adsabs.harvard.edu/abs/1964PhRv..136.1224P
    """
    cc = _GW_DADT_SEP_CONST
    dadt = cc * m1 * m2 * (m1 + m2) / np.power(sma, 3)
    if ecc is not None:
        fe = _gw_ecc_func(ecc)
        dadt *= fe
    return dadt


def gw_lum_circ(mchirp, freq_orb_rest):
    """

    EN07: Eq. 2.2
    """
    lgw_circ = _LUM_CONST * np.power(2.0*np.pi*freq_orb_rest*mchirp, 10.0/3.0)
    return lgw_circ


def gw_strain_source(mchirp, dlum, freq_orb_rest):
    """GW Strain from a single source in a circular orbit.

    e.g. Sesana+2004 Eq.36
    e.g. EN07 Eq.17

    NOTE: THIS IS ORBITAL-FREQUENCY, NOT GW-OBSERVED  [2020-06-17]

    """
    #
    hs = _GW_SRC_CONST * mchirp * np.power(2*mchirp*freq_orb_rest, 2/3) / dlum
    return hs


'''
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

'''
def gw_strain_source_circ(mchirp, dist_lum, freq_orb_rest):
    """GW Strain from a single source in a circular orbit.
    """
    cc = _GW_SRC_CONST
    hs = cc * mchirp * np.power(mchirp*freq_orb_rest, 2/3) / dist_lum
    return hs
'''


def sep_to_merge_in_time(m1, m2, time):
    """The initial separation required to merge within the given time.

    See: [Peters 1964].
    """
    GW_CONST = 64*np.power(NWTG, 3.0)/(5.0*np.power(SPLC, 5.0))
    a1 = astro_core.rad_isco(m1, m2)
    return np.power(GW_CONST*m1*m2*(m1+m2)*time - np.power(a1, 4.0), 1./4.)


def time_to_merge_at_sep(m1, m2, sep):
    """The time required to merge starting from the given initial separation.

    See: [Peters 1964].
    """
    GW_CONST = 64*np.power(NWTG, 3.0)/(5.0*np.power(SPLC, 5.0))
    a1 = astro_core.rad_isco(m1, m2)
    delta_sep = np.power(sep, 4.0) - np.power(a1, 4.0)
    return delta_sep/(GW_CONST*m1*m2*(m1+m2))


def _gw_ecc_func(ecc):
    """GW Hardening rate eccentricitiy dependence F(e).

    See Peters 1964, Eq. 5.6
    EN07: Eq. 2.3
    """
    e2 = ecc*ecc
    num = 1 + (73/24)*e2 + (37/96)*e2*e2
    den = np.power(1 - e2, 7/2)
    fe = num / den
    return fe
