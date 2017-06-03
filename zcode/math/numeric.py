"""General functions for mathematical and numerical operations.

Functions
---------
-   spline                   - Create a general spline interpolation function.
-   cumtrapz_loglog          - Perform a cumulative integral in log-log space.
-   extend                   - Extend the given array by extraplation.
-   sampleInverse            - Find x-sampling to evenly divide a function in y-space.
-   smooth                   - Use convolution to smooth the given array.

-   _trapezium_loglog        -

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import warnings

from . import math_core

__all__ = ['spline', 'cumtrapz_loglog', 'extend', 'sampleInverse', 'smooth']


def spline(xx, yy, order=3, log=True, mono=False, extrap=True, pos=False, sort=True):
    """Create a general, callable spline interpolation function.

    Arguments
    ---------
    xx : (N,), array_like scalars
        Independent variable, must be monotonically increasing -- which `sort`, if `True`, will do.
    yy : (N,), array_like scalars
        Dependent variable; the values of the function.
    order : int
        Order of interpolation (must be 3` if `mono`).
    log : bool
        Interpolate in log-log-space.
    mono : bool
        Use an explicitly monotonic interpolator (`scipy.interpolate.PchipInterpolator`).
    extrap : bool
        Allow extrapolation outside of range of `xx`.
    pos : bool
        Filter to only positive values of `yy` (and corresponding `xx`).
    sort : bool
        Sort the input arrays to assure `xx` is monotonically increasing.

    Returns
    -------
    spline : obj, callable function
        Spline interplation function.

    """
    import scipy.interpolate as sp_interp

    xp = np.array(xx)
    yp = np.array(yy)

    # Make sure arguments are sorted (by independent variable `xx`)
    if sort:
        inds = np.argsort(xp)
        xp = xp[inds]
        yp = yp[inds]

    # Select positive y-values
    if pos:
        inds = np.where(yp > 0.0)[0]
        xp = xp[inds]
        yp = yp[inds]

    # Convert to log-space as needed
    if log:
        xp = np.log10(xp)
        yp = np.log10(yp)

    # Sort input arrays
    inds = np.argsort(xp)
    xp = xp[inds]
    yp = yp[inds]

    # Monotonic Interpolation
    if mono:
        if order != 3: warnings.warn("monotonic `PchipInterpolator` is always cubic!")
        terp = sp_interp.interpolate.PchipInterpolator(xp, yp, extrapolate=extrap)
    # General Interpolation
    else:
        # Let function extrapolate outside range
        if extrap: ext = 0
        # Return zero outside of range
        else:      ext = 1
        terp = sp_interp.interpolate.InterpolatedUnivariateSpline(xp, yp, k=order, ext=ext)

    # Convert back to normal space, as needed
    if log: spline = lambda xx, terp=terp: np.power(10.0, terp(np.log10(xx)))
    else:   spline = terp

    return spline


def cumtrapz_loglog(yy, xx, init=0.0, rev=False):
    """Perform a cumulative integral in log-log space.
    From Thomas Robitaille
    https://github.com/astrofrog/fortranlib/blob/master/src/lib_array.f90
    """
    if(np.ndim(yy) > 1): raise RuntimeError("This isn't implemented for ndim > 1!")

    nums = len(xx)
    sum = np.zeros(nums)

    if(rev):
        xx = xx[::-1]
        yy = yy[::-1]

    sum[0] = init
    for ii in range(1, nums):
        sum[ii] = sum[ii-1] + _trapezium_loglog(xx[ii-1], yy[ii-1], xx[ii], yy[ii])

    if(rev): sum = sum[::-1]

    return sum


def _trapezium_loglog(x1, y1, x2, y2):
    """
    From Thomas Robitaille
    https://github.com/astrofrog/fortranlib/blob/master/src/lib_array.f90
    """
    b = np.log10(y1/y2) / np.log10(x1/x2)
    if(np.fabs(b+1.0) < 1.0e-10):
        trap = x1 * y1 * np.log(x2/x1)
    else:
        trap = y1 * (x2*(x2/x1)**b-x1) / (b+1.0)

    return trap


def extend(arr, num=1, log=True, append=False):
    """Extend the given array by extraplation.

    Arguments
    ---------
        arr    <flt>[N] : array to extend
        num    <int>    : number of points to add (on each side, if ``both``)
        log    <bool>   : extrapolate in log-space
        append <bool>   : add the extended points onto the given array

    Returns
    -------
        retval <flt>[M] : extension (or input ``arr`` with extension added, if ``append``).

    """

    if(log): useArr = np.log10(arr)
    else:      useArr = np.array(arr)

    steps = np.arange(1, num+1)
    left = useArr[0] + (useArr[0] - useArr[1])*steps[::-1].squeeze()
    rigt = useArr[-1] + (useArr[-1] - useArr[-2])*steps.squeeze()

    if(log):
        left = np.power(10.0, left)
        rigt = np.power(10.0, rigt)

    if(append): return np.hstack([left, arr, rigt])
    return [left, rigt]


def sampleInverse(xx, yy, num=100, log=True, sort=False):
    """Find the x-sampling of a function to evenly divide its results in y-space.

    Input function *must* be strictly monotonic in ``yy``.

    Arguments
    ---------
        xx   <flt>[N] : array(scalar), initial sample space
        yy   <flt>[N] : function to resample
        num  <int>    : number of points to produce
        log  <bool>   : sample in log space
        sort <bool>   : sort return array ``samps``

    Returns
    -------
        samps <flt>[``num``] : new sample points from ``xx``

    """

    # Convert to log-space, as needed
    if log:
        xp = np.log10(xx)
        yp = np.log10(yy)
    else:
        xp = np.array(xx)
        yp = np.array(yy)

    inds = np.argsort(yp)
    xp = xp[inds]
    yp = yp[inds]

    # Construct Interpolating Function, *must be monotonic*
    interpBack = spline(yp, xp, log=False, mono=True)

    # Divide y-axis evenly, and find corresponding x-points
    #     Note: `log` spacing is enforced manually, use `lin` here!
    levels = math_core.spacing(yp, scale='lin', num=num)
    samples = interpBack(levels)

    # Convert back to normal space, as needed
    if log: samples = np.power(10.0, samples)

    if sort: samples = samples[np.argsort(samples)]

    return samples


def smooth(arr, size, width=None, loc=None, mode='same'):
    """Use convolution to smooth the given array.

    The ``width``, ``loc`` and ``size`` arguments can be given as integers, in which case they are taken
    as indices in the input array; or they can be floats, in which case they are interpreted as
    fractions of the length of the input array.

    Arguments
    ---------
    arr   <flt>[N] : input array to be smoothed
    size  <obj>    : size of smoothing window
    width <obj>    : scalar specifying the region to be smoothed, if two values are given
                     they are taken as left and right bounds
    loc   <flt>    : int or float specifying to center position of smoothing,
                     ``width`` is used relative to this position, if provided.
    mode  <str>    : type of convolution, passed to ``numpy.convolve``

    Returns
    -------
    smArr <flt>[N] : smoothed array

    """

    length = np.size(arr)
    size = math_core._fracToInt(size, length, within=1.0, round='floor')

    assert size <= length, "`size` must be less than length of input array!"

    window = np.ones(int(size))/float(size)

    # Smooth entire array
    smArr = np.convolve(arr, window, mode=mode)

    # Return full smoothed array if no bounds given
    if width is None:
        return smArr

    # Other convolution modes require dealing with differing lengths
    #    If smoothing only a portion of the array,
    assert mode == 'same', "Other convolution modes not supported for portions of array!"

    # Smooth portion of array
    # -----------------------

    if np.size(width) == 2:
        lef = width[0]
        rit = width[1]
    elif np.size(width) == 1:
        if loc is None:
            raise ValueError("For a singular ``width``, ``pos`` must be provided!")
        lef = width
        rit = width
    else:
        raise ValueError("``width`` must be one or two scalars!")

    # Convert fractions to positions, if needed
    lef = math_core._fracToInt(lef, length-1, within=1.0, round='floor')
    rit = math_core._fracToInt(rit, length-1, within=1.0, round='floor')

    # If ``loc`` is provided, use ``width`` relative to that
    if loc is not None:
        loc = math_core._fracToInt(loc, length-1, within=1.0, round='floor')
        lef = loc - lef
        rit = loc + rit

    mask = np.ones(length, dtype=bool)
    mask[lef:rit] = False
    smArr[mask] = arr[mask]

    return smArr
