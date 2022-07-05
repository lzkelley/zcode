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
# import scipy as sp
import scipy.stats  # noqa
import warnings

from . import math_core, interpolate  # , statistic

__all__ = [
    'cumtrapz_loglog', 'even_selection', 'extend', 'monotonic_smooth', 'rk4_step',
    'sample_inverse', 'smooth_convolve', 'spline',
    # DEPRECATED
    'smooth', '_smooth'
]


def cumtrapz_loglog(yy, xx, bounds=None, axis=-1, dlogx=None, lntol=1e-2):
    """Calculate integral, given `y = dA/dx` or `y = dA/dlogx` w/ trapezoid rule in log-log space.

    We are calculating the integral `A` given sets of values for `y` and `x`.
    To associate `yy` with `dA/dx` then `dlogx = None` [default], otherwise,
    to associate `yy` with `dA/dlogx` then `dlogx = True` for natural-logarithm, or `dlogx = b`
    for a logarithm of base `b`.

    For each interval (x[i+1], x[i]), calculate the integral assuming that y is of the form,
        `y = a * x^gamma`

    Notes
    -----
    - When bounds are given that are not identical to input `xx` values, then interpolation must
      be performed.  This can be done on the resulting cumsum'd values, or on the input integrand
      values.  The cumsum values are *not necessarily a power-law* (for negative indices), and thus
      the interpolation is better performed on the input `yy` values.

    """
    yy = np.asarray(yy)
    xx = np.asarray(xx)

    if bounds is not None:
        if len(bounds) != 2 or np.any(~math_core.within(bounds, xx)) or (bounds[0] > bounds[1]):
            err = "Invalid `bounds` = '{}', xx extrema = '{}'!".format(
                bounds, math_core.minmax(xx))
            raise ValueError(err)

        if axis != -1 or np.ndim(yy) > 1:
            newy = interpolate.interp_func(xx, yy, xlog=True, ylog=True)(bounds)
        else:
            newy = interpolate.interp(bounds, xx, yy, xlog=True, ylog=True, valid=False)

        # newy = interpolate.interp(bounds, xx, yy, xlog=True, ylog=True, valid=False)
        ii = np.searchsorted(xx, bounds)
        xx = np.insert(xx, ii, bounds, axis=axis)
        yy = np.insert(yy, ii, newy, axis=axis)
        ii = np.array([ii[0], ii[1]+1])
        assert np.alltrue(xx[ii] == bounds), "FAILED!"

    yy = np.ma.masked_values(yy, value=0.0, atol=0.0)

    ''' NOTE: this doesn't work if `xx` is expanded, but `axis != 0`
    # if np.ndim(yy) > 1 and np.ndim(xx) == 1:
    if np.ndim(yy) != np.ndim(xx):
        if np.ndim(yy) < np.ndim(xx):
            raise ValueError("BAD SHAPES")
        cut = [slice(None)] + [np.newaxis for ii in range(np.ndim(yy)-1)]
        xx = xx[tuple(cut)]
    '''

    if np.ndim(yy) != np.ndim(xx):
        if np.ndim(yy) > 1 and np.ndim(xx) > 1:
            raise ValueError(f"{np.ndim(yy)=}, {np.ndim(xx)=} || provide either a 1D `xx` or the correct shape!")
        if np.ndim(yy) < np.ndim(xx):
            raise ValueError("BAD SHAPES")
        # This only works if `ndim(xx) == 1`
        cut = [slice(None)] + [np.newaxis for ii in range(np.ndim(yy)-1)]
        xx = xx[tuple(cut)]
        xx = np.moveaxis(xx, 0, axis)

    if np.shape(xx)[axis] != np.shape(yy)[axis]:
        raise ValueError(f"Shape mismatch!  {np.shape(xx)=} {np.shape(yy)=} | {axis=}")

    log_base = np.e
    if dlogx is not None:
        # If `dlogx` is True, then we're using log-base-e (i.e. natural-log)
        # Otherwise, set the log-base to the given value
        if dlogx is not True:
            log_base = dlogx

    # Numerically calculate the local power-law index
    delta_logx = np.diff(np.log(xx), axis=axis)
    gamma = np.diff(np.log(yy), axis=axis) / delta_logx
    xx = np.moveaxis(xx, axis, 0)
    yy = np.moveaxis(yy, axis, 0)
    aa = np.mean([xx[:-1] * yy[:-1], xx[1:] * yy[1:]], axis=0)
    aa = np.moveaxis(aa, 0, axis)
    xx = np.moveaxis(xx, 0, axis)
    yy = np.moveaxis(yy, 0, axis)
    # Integrate dA/dx
    # A = (x1*y1 - x0*y0) / (gamma + 1)
    if dlogx is None:
        dz = np.diff(yy * xx, axis=axis)
        trapz = dz / (gamma + 1)
        # when the power-law is (near) '-1' then, `A = a * log(x1/x0)`
        idx = np.isclose(gamma, -1.0, atol=lntol, rtol=lntol)

    # Integrate dA/dlogx
    # A = (y1 - y0) / gamma
    else:
        dy = np.diff(yy, axis=axis)
        trapz = dy / gamma
        # when the power-law is (near) '-1' then, `A = a * log(x1/x0)`
        idx = np.isclose(gamma, 0.0, atol=lntol, rtol=lntol)

    trapz[idx] = aa[idx] * delta_logx[idx]

    integ = np.log(log_base) * np.cumsum(trapz, axis=axis)
    if bounds is not None:
        # NOTE: **DO NOT INTERPOLATE INTEGRAL** this works badly for negative power-laws
        # lo, hi = interpolate.interp(bounds, xx[1:], integ, xlog=True, ylog=True, valid=False)
        # integ = hi - lo
        integ = np.moveaxis(integ, axis, 0)
        lo, hi = integ[ii-1, ...]
        integ = hi - lo

    return integ


def even_selection(size, select, sel_is_true=True, return_indices=False):
    """Create a boolean indexing array of length `size` with `select`, evenly spaced elements.

    Arguments
    ---------
    size : int,
        Total size of array.
    select : int,
        Number of entries to select.
    sel_is_true : bool,
        Whether the 'selected' elements should be designated as `True` or `False`.
        * `sel_is_true == True`  then there are `select` True  elements and the rest are False.
        * `sel_is_true == False` then there are `select` False elements and the rest are True.
    return_indices : bool,
        Return array index numbers instead of a boolean array.  Uses `numpy.where` and may be slow.
        * True : return array index numbers (with `select` elements)
        * False : return boolean array (with `size` elements)

    Returns
    -------
    cut : array

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
        idx = [q*i + min(i, r) for i in range(size-select)]
        cut[idx] = n
    else:
        cut = np.ones(size, dtype=bool) * n
        q, r = divmod(size, select)
        idx = [q*i + min(i, r) for i in range(select)]
        cut[idx] = y

    if return_indices:
        cut = np.where(cut)[0]

    return cut


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

    useArr = np.log10(arr) if log else np.asarray(arr)

    steps = np.arange(1, num+1)
    left = useArr[0] + (useArr[0] - useArr[1])*steps[::-1].squeeze()
    rigt = useArr[-1] + (useArr[-1] - useArr[-2])*steps.squeeze()

    if log:
        left = np.power(10.0, left)
        rigt = np.power(10.0, rigt)

    if append:
        return np.hstack([left, arr, rigt])

    return [left, rigt]


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


def ndinterp(xx, xvals, yvals, xlog=True, ylog=True):
    """Interpolate 2D data to an array of points.

    `xvals` and `yvals` are (N, M) where the interpolation is done along the 1th (`M`)
    axis (i.e. interpolation is done independently for each `N` row.  Should be generalizeable to
    higher dim.

    """
    # Convert to (N, T, M)
    #     `xx` is (T,)  `xvals` is (N, M) for N-binaries and M-steps
    select = (xx[np.newaxis, :, np.newaxis] <= xvals[:, np.newaxis, :])

    # (N, T)
    aft = np.argmax(select, axis=-1)
    # zero values in `aft` mean no xvals after the targets were found
    valid = (aft > 0)
    inval = ~valid
    bef = np.copy(aft)
    bef[valid] -= 1

    # (2, N, T)
    cut = [aft, bef]
    # (2, N, T)
    xvals = [np.take_along_axis(xvals, cc, axis=-1) for cc in cut]
    # Find how far to interpolate between values (in log-space)
    #     (N, T)
    frac = (xx[np.newaxis, :] - xvals[1]) / np.subtract(*xvals)

    # (2, N, T)
    data = [np.take_along_axis(yvals, cc, axis=-1) for cc in cut]
    # Interpolate by `frac` for each binary
    new = data[1] + (np.subtract(*data) * frac)
    # Set invalid binaries to nan
    new[inval, ...] = np.nan
    new = new
    return new


def regress(xx, yy):
    """Perform *linear* regression on the *zeroth* dimension of the given (ND) data.

    Arguments
    ---------
    xx : (N, ...) array_like of scalar
        Independent variable of data.
    yy : (N, ...) array_like of scalar
        Dependent variable of data, with shape matching that of `xx`.

    Returns
    -------
    coeff : (2, ...) np.ndarray of scalar
        The linear regression coefficients, such that the 0th element is the slope, and the 1st is
        the y-intercept.  The shape of `coeff` is such that ``coeff.shape[1:] == xx.shape[1:]``.
    zz : (N, ...) np.ndarray of scalar
        The model/prediction values using the linear regression and the input `xx` values.
        Same shape as `xx` and `yy`.

    """
    if np.shape(xx) != np.shape(yy):
        err = "Shape of xx ({}) does not match that of yy ({})!".format(np.shape(xx), np.shape(yy))
        raise ValueError(err)

    # print("\n=REGRESS=")
    # print(f"{xx.shape=}, {yy.shape=}")
    aa = np.concatenate([xx[np.newaxis, :], np.ones_like(xx)[np.newaxis, :]])
    # print(f"{aa.shape=}")
    # Calculate A^T (transpose of `aa`) times `aa` and times `yy`
    ata = np.einsum('ji...,ki...->jk...', aa, aa)
    aty = np.einsum('ji...,i...->j...', aa, yy)
    # print(f"{ata.shape=}, {aty.shape=}")

    # Find the inverse of A^T (move the 2 axes for inversion to end, required by `np.linalg.inv`)
    ata = np.moveaxis(ata, 0, -1)
    ata = np.moveaxis(ata, 0, -1)
    ata_inv = np.linalg.inv(ata)
    # print(f"{ata_inv.shape=}")
    # Move the inverted axes back to the front
    ata_inv = np.moveaxis(ata_inv, -1, 0)
    ata_inv = np.moveaxis(ata_inv, -1, 0)

    # Solve for the regression coefficients
    # coeff = (A^T A)^-1 * (A^T y)
    coeff = np.einsum('ij...,j...->i...', ata_inv, aty)
    # Calculate the predicted/modeled y-values based on the regression coefficients
    zz = np.einsum('ji...,j...->i...', aa, coeff)
    return coeff, zz


def rk4_step(func, x0, y0, dx, args=None, check_nan=0, check_nan_max=5, debug=False):
    if args is None:
        k1 = dx * func(x0, y0)
        k2 = dx * func(x0 + dx/2.0, y0 + k1/2.0)
        k3 = dx * func(x0 + dx/2.0, y0 + k2/2.0)
        k4 = dx * func(x0 + dx, y0 + k3)
    else:
        k1 = dx * func(x0, y0, *args)
        k2 = dx * func(x0 + dx/2.0, y0 + k1/2.0, *args)
        k3 = dx * func(x0 + dx/2.0, y0 + k2/2.0, *args)
        k4 = dx * func(x0 + dx, y0 + k3, *args)

    y1 = y0 + (1.0/6.0) * (k1 + 2*k2 + 2*k3 + k4)
    x1 = x0 + dx

    if debug:
        xs = [x0, x0 + dx/2, x0 + dx/2, x0 + dx]
        ys = [y0, y0 + k1/2, y0 + k2/2, y0 + k3]
        ks = [k1, k2, k3, k4]
        for ii, (_x, _y, _k) in enumerate(zip(xs, ys, ks)):
            print("\t{} {:.4e} {:.4e} {:.4e}".format(ii+1, _x, _y, _k/dx))

    # Try recursively decreasing step-size until finite-value is reached
    if check_nan > 0 and not np.isfinite(y1):
        '''
        xvals = [x0, x0 + dx/2.0, x0 + dx/2.0, x0 + dx]
        yvals = [y0, y0 + k1/2.0, y0 + k2/2.0, y0 + k3]
        kvals = [k1, k2, k3, k4]
        for ii in range(4):
            print("\t"*check_nan, ii, dx, xvals[ii], yvals[ii], kvals[ii])
        '''

        if check_nan > check_nan_max:
            err = "Failed to find finite step!  `check_nan` = {}!".format(check_nan)
            raise RuntimeError(err)
        # Note that `True+1 = 2`
        rk4_step(func, x0, y0, dx/2.0, check_nan=check_nan+1, check_nan_max=check_nan_max)

    # xvals = [x0, x0 + dx/2, x0 + dx/2, x0 + dx]
    # dys = [1.0, 0.5, 0.5, 1.0]
    # yn = y0
    # prev = 0.0
    # for ii, (xv, dy) in enumerate(zip(xvals, dys)):
    #     yv = y0 + prev * dy       # [0.0, k1/2, k2/2, k3]
    #     ki = dx * func(xv, yv)
    #     yn += (ki / dy) / 6.0   # [k1, 2*k2, 2*k3, k4] / 6
    #     prev = ki
    #
    # xn = x0 + dx
    # return xn, yn

    return x1, y1


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
    NOTE: length(output) != length(input), to correct this:
          return y[(window_len/2-1):-(window_len/2)] instead of just y.

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


def _log_from_scale(log, scale):
    if log is None:
        if scale is None:
            log = True
            scale = 'log'
        elif scale.startswith('log'):
            log = True
        elif scale.startswith('linear'):
            log = False
        else:
            raise ValueError("Unrecognized `scale` parameter '{}'".format(scale))

    if scale is None:
        if log is True:
            scale = 'log'
        elif log is False:
            scale = 'linear'
        else:
            raise ValueError("Unrecognized `log` parameter '{}'".format(log))

    return log, scale


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


'''
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
'''
