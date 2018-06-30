"""Common astrophysical scaling relations.
"""

import numpy as np

__all__ = ['mbh_sigma', 'mbh_sigma_inv']

from zcode.constants import MSOL


def mbh_sigma(sigma):
    """McConnell + Ma 2012 [1211.2816]

    See Abstract:
    http://adsabs.harvard.edu/abs/2013ApJ...764..184M

    Arguments
    ---------
    sigma : array_like scalar [cm/s]
        Velocity dispersion in [centimeters per second]

    Returns
    -------
    mbh : array_like scalar [g]
        Black-hole mass in [grams]

    """
    _MASS = 8.32
    _SLOPE = 5.64
    _VEL = 200e5

    mbh = _MASS + _SLOPE*np.log10(sigma/_VEL)
    mbh = np.power(10.0, mbh) * MSOL
    return mbh


def mbh_sigma_inv(mbh):
    """McConnell + Ma 2012 [1211.2816]

    See Abstract:
    http://adsabs.harvard.edu/abs/2013ApJ...764..184M

    Arguments
    ---------
    mbh : array_like scalar [g]
        Black-hole mass in [grams]

    Returns
    -------
    sigma : array_like scalar [cm/s]
        Velocity dispersion in [centimeters per second]

    """
    _MASS = 8.32
    _SLOPE = 5.64
    _VEL = 200e5

    sigma = (np.log10(mbh/MSOL) - _MASS) / _SLOPE
    sigma = np.power(10.0, sigma) * _VEL
    return sigma
