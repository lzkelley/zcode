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
from .. import utils

__all__ = [
    'cumtrapz_loglog', 'even_selection', 'extend', 'monotonic_smooth', 'sample_inverse',
    'smooth_convolve', 'spline',
    # DEPRECATED
    'sampleInverse', 'smooth', '_smooth'
]


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
        if order != 3:
            warnings.warn("monotonic `PchipInterpolator` is always cubic!")
        terp = sp_interp.PchipInterpolator(xp, yp, extrapolate=extrap)
    # General Interpolation
    else:
        # Let function extrapolate outside range
        if extrap:
            ext = 0
        # Return zero outside of range
        else:
            ext = 1
        terp = sp_interp.InterpolatedUnivariateSpline(xp, yp, k=order, ext=ext)

    # Convert back to normal space, as needed
    if log:
        spline = lambda xx, terp=terp: np.power(10.0, terp(np.log10(xx)))
    else:
        spline = terp

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


def sampleInverse(*args, **kwargs):
    utils.dep_warn("sampleInverse", newname="sample_inverse")
    return sample_inverse(*args, **kwargs)


def sample_inverse(xx, yy, num=100, log=True, sort=False):
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
    interp_back = spline(yp, xp, log=False, mono=True)

    # Divide y-axis evenly, and find corresponding x-points
    #     Note: `log` spacing is enforced manually, use `lin` here!
    levels = math_core.spacing(yp, scale='lin', num=num)
    samples = interp_back(levels)

    # Convert back to normal space, as needed
    if log:
        samples = np.power(10.0, samples)

    if sort:
        samples = samples[np.argsort(samples)]

    return samples


def smooth_convolve(vals, window_size=10, window='hanning'):
    """smooth the data using a window with requested size.

    NOTE: taken from http://scipy-cookbook.readthedocs.io/items/SignalSmooth.html

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t = linspace(-2, 2, 0.1)
    x = sin(t) + randn(len(t)) * 0.1
    y = smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """
    _valid_windows = ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']

    if vals.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if vals.size < window_size:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_size < 3:
        return vals

    if window not in _valid_windows:
        raise ValueError("Window is not one of {}".format(_valid_windows))

    s = np.r_[vals[window_size-1:0:-1], vals, vals[-2:-window_size-1:-1]]
    # moving average
    if window == 'flat':
        w = np.ones(window_size, 'd')
    else:
        # w = eval('numpy.' + window + '(window_len)')
        w = getattr(np, window)(window_size)

    y = np.convolve(w/w.sum(), s, mode='valid')
    # Fix issue with
    lo = (window_size/2-1)
    hi = (window_size/2)
    lo = int(np.ceil(lo))
    hi = int(np.floor(hi))
    y = y[lo:-hi]
    return y


def monotonic_smooth(vals, window_size=4, expand_size=1, max_iter=10,
                     thresh=-0.01, details=False, **kwargs):
    """Try to smooth out non-monotonicities in the given array.

    NOTE: causes some sub-optimal edge effects.

    Arguments
    ---------
    vals: (N,) scalar
        Input values to smooth.
    window_size
    expand_size: int,
        Expand the region being smoothed over to include this many neighboring points,
        outside of the non-monotonic region.
    max_iter
    thresh : scalar,
        Differences between subsequent points must be less than this value to be considered
        as non-monotonicities.

    Returns
    -------
    yy: (N,) scalar
        Smoothed array.

    """
    if np.ndim(vals) > 1:
        raise ValueError("Input array must be 1D")

    yy = np.copy(vals)

    # Smooth if the curve is not monotonic
    bads = (np.diff(yy) < thresh)
    cnt = 0
    dets = []
    while any(bads) and cnt < max_iter:
        bads = np.where(bads)[0]
        lo = bads[0]
        lo = np.max([lo - expand_size, 0])
        hi = bads[-1]+1
        hi = np.min([hi + expand_size, yy.size + 1])
        if details:
            dets.append([[lo, hi], np.copy(yy[lo:hi])])

        yy[lo:hi] = smooth_convolve(yy, window_size, **kwargs)[lo:hi]
        bads = (np.diff(yy) < thresh)
        cnt += 1

    if details:
        return yy, dets

    return yy


def even_selection(size, select, sel_is_true=True):
    """Create a boolean indexing array of length `size` with `select`, evenly spaced elements.

    If `sel_is_true == True`  then there are `select` True  elements and the rest are False.
    If `sel_is_true == False` then there are `select` False elements and the rest are True.

    """
    y = True if sel_is_true else False
    n = (not y)

    if select > size:
        raise ValueError("Cannot select {}/{} elements!".format(select, size))

    if select == size:
        cut = np.ones(size, dtype=bool) * y
    elif select > size/2:
        cut = np.ones(size, dtype=bool) * y
        q, r = divmod(size, size-select)
        indices = [q*i + min(i, r) for i in range(size-select)]
        cut[indices] = n
    else:
        cut = np.ones(size, dtype=bool) * n
        q, r = divmod(size, select)
        indices = [q*i + min(i, r) for i in range(select)]
        cut[indices] = y

    return cut


# ==================================================
# ===============    DEPRECATED    =================
# ==================================================


def smooth(*args, **kwargs):
    warnings.warn("The `zcode.math.numeric.smooth` function is being deprecated!",
                  DeprecationWarning, stacklevel=3)
    return _smooth(*args, **kwargs)


def _smooth(arr, size, width=None, loc=None, mode='same'):
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
