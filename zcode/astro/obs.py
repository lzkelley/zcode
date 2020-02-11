"""Astronomical observations calculations.
"""

import numpy as np

from zcode.constants import PC, SPLC

# VEGA/Johnson/Bessell: http://web.ipac.caltech.edu/staff/fmasci/home/astro_refs/magsystems.pdf
# SDSS/AB/Fukugita: http://www.astronomy.ohio-state.edu/~martini/usefuldata.html

# These wavelengths are in [cm]
BAND_EFF_LOC = {
    # Vega/Johnson/Bessell
    "U": {"l": 366e-7},
    "B": {"l": 438e-7},
    "V": {"l": 545e-7},
    "R": {"l": 641e-7},
    "I": {"l": 798e-7},
    # SDSS AB Magnitudes
    "u": {"l": 356e-7},
    "g": {"l": 483e-7},
    "r": {"l": 626e-7},
    "i": {"l": 767e-7},
    "z": {"l": 910e-7}
}
BAND_REF_FLUX = {
    # Vega/Johnson/Bessell
    "U": {"f": 1.790, "l": 417.5},
    "B": {"f": 4.063, "l": 632.0},
    "V": {"f": 2.636, "l": 363.1},
    "R": {"f": 3.064, "l": 217.7},
    "I": {"f": 2.416, "l": 112.6},
    # SDSS AB Magnitudes
    "u": {"f": 3.631, "l": 859.5},
    "g": {"f": 3.631, "l": 466.9},
    "r": {"f": 3.631, "l": 278.0},
    "i": {"f": 3.631, "l": 185.2},
    "z": {"f": 3.631, "l": 131.5}
}
BAND_ZERO_POINT = {
    # Vega/Johnson/Bessell
    "U": {"f": +0.770, "l": -0.152},
    "B": {"f": -0.120, "l": -0.602},
    "V": {"f": +0.000, "l": +0.000},
    "R": {"f": +0.186, "l": +0.555},
    "I": {"f": +0.444, "l": +1.271},
    # SDSS AB Magnitudes
    "u": {"f": 0.0, "l": 0.0},
    "g": {"f": 0.0, "l": 0.0},
    "r": {"f": 0.0, "l": 0.0},
    "i": {"f": 0.0, "l": 0.0},
    "z": {"f": 0.0, "l": 0.0}
}
UNITS = {
    "f": 1.0e-20,  # erg/s/Hz/cm^2
    "l": 1.0e-11   # erg/s/Angstrom/cm^2
}

__all__ = ["ABmag_to_flux", "abs_mag_to_lum", "flux_to_mag", "lum_to_abs_mag", "mag_to_flux",
           "fnu_to_flambda", "flambda_to_fnu"]


# _band_name = ['u', 'b', 'v', 'r', 'i']
# _band_wlen = [365, 445, 551, 658, 806]   # nm
# _band_color = ['violet', 'blue', 'green', 'red', 'darkred']
# Band = namedtuple('band', ['name', 'freq', 'wlen', 'color'])
#
# BANDS = {nn: Band(nn, SPLC/(ll*1e-7), ll*1e-7, cc)
#          for nn, ll, cc in zip(_band_name, _band_wlen, _band_color)}


def _get_units_type(type):
    try:
        units = UNITS[type]
    except Exception as err:
        raise ValueError("Unrecognized `type` = '{}'".format(type))

    return units, type


def ABmag_to_flux(mag):
    """Convert from AB Magnitude to spectral-flux density.

    See: http://web.ipac.caltech.edu/staff/fmasci/home/astro_refs/magsystems.pdf

    Returns
    -------
    fnu : () scalar
        Spectral-flux density in units of [erg/s/cm^2/Hz]

    """
    fnu = np.power(10.0, (mag + 48.6)/-2.5)
    return fnu


def mag_to_flux(band, mag, type='f'):
    """Convert from broad-band filter magnitude (e.g. Johnson) to flux.

    Returns
    -------
    flux : () scalar
        Flux in either [erg/s/cm^2/Hz] or [erg/s/cm^2/Angstrom] depending on `type`.

    """
    mag = np.asarray(mag)
    units, type = _get_units_type(type)

    if band not in BAND_REF_FLUX.keys():
        raise ValueError("Unrecognized `band` = '{}'".format(band))

    ref_flux = BAND_REF_FLUX[band][type] * units
    # zero_point = BAND_ZERO_POINT[band][type]
    flux = ref_flux * np.power(10.0, mag/-2.5)
    return flux


def flux_to_mag(band, flux, type='f'):
    """Convert from broad-band filter magnitude (e.g. Johnson) to flux.

    Arguments
    ---------
    band
    flux : () scalar
        Flux in either [erg/s/cm^2/Hz] or [erg/s/cm^2/Angstrom] depending on `type`.
    type

    Returns
    -------
    mag

    """
    flux = np.asarray(flux)
    units, type = _get_units_type(type)

    if band not in BAND_REF_FLUX.keys():
        raise ValueError("Unrecognized `band` = '{}'".format(band))

    ref_flux = BAND_REF_FLUX[band][type] * units
    # zero_point = BAND_ZERO_POINT[band][type]
    # flux = ref_flux * np.power(10.0, mag/-2.5)

    mag = -2.5 * np.log10(flux/ref_flux)

    return mag


def abs_mag_to_lum(band, mag, type='f'):
    mag = np.asarray(mag)
    if type.lower().startswith('f'):
        type = 'f'
        units = 1.0e-20   # erg/s/Hz/cm^2
    elif (type.lower().startswith('l') or type.lower().startswith('w')):
        type = 'l'
        units = 1.0e-11   # erg/s/Angstrom/cm^2
    else:
        raise ValueError("Unrecognized `type` = '{}'".format(type))

    if band not in BAND_REF_FLUX.keys():
        raise ValueError("Unrecognized `band` = '{}'".format(band))

    ref_flux = BAND_REF_FLUX[band][type]
    lum = 4.0 * np.pi * ref_flux * units * PC**2 * np.power(10.0, 2-mag/2.5)
    return lum


def lum_to_abs_mag(band, lum, type='f'):
    lum = np.asarray(lum)
    if type.lower().startswith('f'):
        type = 'f'
        units = 1.0e-20   # erg/s/Hz/cm^2
    elif (type.lower().startswith('l') or type.lower().startswith('w')):
        type = 'l'
        units = 1.0e-11   # erg/s/Angstrom/cm^2
    else:
        raise ValueError("Unrecognized `type` = '{}'".format(type))

    if band not in BAND_REF_FLUX.keys():
        raise ValueError("Unrecognized `band` = '{}'".format(band))

    ref_lum = BAND_REF_FLUX[band][type] * 4.0 * np.pi * units * (10*PC)**2
    mag = lum/ref_lum
    mag = -2.5 * np.log10(mag) + 5
    return mag


def fnu_to_flambda(fnu, freq=None, wavelength=None):
    if freq is None:
        freq = SPLC / wavelength

    flambda = fnu * freq**2 / SPLC
    return flambda


def flambda_to_fnu(flambda, freq=None, wavelength=None):
    if wavelength is None:
        wavelength = SPLC / freq

    fnu = flambda * freq**2 / SPLC
    return fnu
