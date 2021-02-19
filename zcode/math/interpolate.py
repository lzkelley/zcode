"""
"""

import numpy as np

import scipy as sp
import scipy.interpolate  # noqa


__all__ = [
    "Interp2D_Uniform", "Interp2D_RegIrr",
    'interp', 'interp_axis', 'interp_func',
]

import zcode
import zcode.math


class Interp2D_Uniform:
    """Linear Interpolate/Extrapolate on a Regular/Uniform Grid in 2D.
    """

    def __init__(self, xx, yy, zz, extrap=True):
        dx = np.diff(xx)
        dy = np.diff(yy)
        for nn, dd, vv in zip(['x', 'y'], [dx, dy], [xx, yy]):
            if np.ndim(vv) != 1 or np.size(vv) < 2:
                err = "Input {} array (shape {}) must be 1D and longer than 2 elements".format(
                    nn, np.shape(vv))
                raise ValueError(err)

            if not np.allclose(dd, dd[0], atol=0.0, rtol=1e-8):
                raise ValueError("Spacing between `{}` values is not uniform!  d{} = {}".format(
                    nn, nn, zcode.math.stats_str(dd)))

        if np.shape(zz) != (np.size(xx), np.size(yy)):
            raise ValueError("Shape of zz = {}, must match shape(x, y) = {}, {}".format(
                np.shape(zz), np.size(xx), np.size(yy)))

        self._extrap = extrap

        self._x0 = xx[0]
        self._x1 = xx[-1]
        self._y0 = yy[0]
        self._y1 = yy[-1]
        self._dx = dx[0]
        self._dy = dy[0]
        self._zz = np.array(zz)
        self._shape = self._zz.shape
        return

    def __call__(self, uu, vv, check=True):
        uu = np.atleast_1d(uu)
        vv = np.atleast_1d(vv)

        extrap = self._extrap
        # If extrapolation is not occurring, find values outside of bounds
        if extrap is not True:
            extrema = [[self._x0, self._x1], [self._y0, self._y1]]
            ui = zcode.math.math_core.within(uu, extrema[0], inv=True)
            vi = zcode.math.math_core.within(vv, extrema[1], inv=True)
            # If extrapolation is disallowed, check for exterior values and raise error as needed
            if extrap is False:
                for nn, vals, ii, extr in zip(['uu', 'vv'], [uu, vv], [ui, vi], extrema):
                    if np.any(ii):
                        err = ("Extrapolate False :: `{}` values outside of bounds"
                               " ({:.2e}, {:.2e}) :: {}")
                        err = err.format(nn, *extr, zcode.math.math_core.array_str(vals[ii]))
                        raise ValueError(err)
            # If extertior vals should be filled, make sure `extrap` is a valid scalar, fill later
            elif not np.isscalar(extrap):
                err = "`extrap` ({}) should be True, False, or a float (scalar) value!".format(
                    extrap)
                raise ValueError(err)

        xi = np.floor((uu - self._x0) / self._dx).astype(int)
        yi = np.floor((vv - self._y0) / self._dy).astype(int)
        xi = np.clip(xi, 0, self._shape[0] - 2)
        yi = np.clip(yi, 0, self._shape[1] - 2)

        xlo = xi * self._dx + self._x0
        xhi = (xi + 1) * self._dx + self._x0
        dx_lo = uu - xlo
        dx_hi = xhi - uu
        ylo = yi * self._dy + self._y0
        yhi = (yi + 1) * self._dy + self._y0
        dy_lo = vv - ylo
        dy_hi = yhi - vv

        rv = 0.0
        for ii, dx in enumerate([dx_hi, dx_lo]):
            for jj, dy in enumerate([dy_hi, dy_lo]):
                zz = self._zz[xi + ii, yi + jj]
                rv += zz * dx * dy

        rv /= (self._dx * self._dy)
        if check:
            fails = ~np.isfinite(rv)
            if np.any(fails):
                raise ValueError("Infinite interpolated values!")

        # If extrapolated values should be filled, fill them
        if (extrap is not True) and (extrap is not False):
            ii = ui | vi
            rv[ii] = extrap

        return rv


class Interp2D_RegIrr:
    """
    """

    def __init__(self, xx, yy, zz):
        nams = ['xx', 'yy', 'zz']
        vals = [xx, yy, zz]
        dims = [1, 2, 2]
        for nn, vv, dd in zip(nams, vals, dims):
            if np.ndim(vv) == dd:
                continue

            raise ValueError("Dimension of {} ({}) must be {}!".format(nn, np.ndim(vv), dd))

        nx, ny = yy.shape
        names = ['xx', 'zz']
        vals = [xx, zz]
        shapes = [nx, (nx, ny)]
        for nn, vv, ss in zip(names, vals, shapes):
            if np.shape(vv) == ss:
                continue
            raise ValueError("Shape of {} ({}) must be {} to match shape of yy ({})!".format(
                nn, np.shape(vv), ss, np.shape(yy)))

        dx = np.diff(xx)
        if not np.allclose(dx, dx[0], atol=0.0, rtol=1e-8):
            raise ValueError("Spacing between `xx` values is not uniform!  dx = {}".format(
                nn, zcode.math.stats_str(dx)))

        if not np.all(np.diff(yy, axis=-1) > 0.0):
            raise ValueError("All `yy` values must be ascending in the 1th dimension!")

        self._x0 = xx[0]
        self._x1 = xx[-1]
        self._dx = dx[0]

        self._yy = np.array(yy)
        self._zz = np.array(zz)
        self._shape = self._zz.shape
        return

    def __call__(self, uu, vv):
        uu = np.atleast_1d(uu)
        vv = np.atleast_1d(vv)

        xi = np.floor((uu - self._x0) / self._dx).astype(int)
        xi = np.clip(xi, 0, self._shape[0] - 2)

        xlo = xi * self._dx + self._x0
        xhi = (xi + 1) * self._dx + self._x0
        dx_lo = uu - xlo
        dx_hi = xhi - uu
        for ai, dal, dah, bb in zip(xi, dx_lo, dx_hi, vv):
            newy = self._yy[ai, :] * dal + self._yy[ai+1, :] * dah
            newy /= self._dx

        ylo = yi * self._dy + self._y0
        yhi = (yi + 1) * self._dy + self._y0
        dy_lo = vv - ylo
        dy_hi = yhi - vv

        rv = 0.0
        for ii, dx in enumerate([dx_hi, dx_lo]):
            for jj, dy in enumerate([dy_hi, dy_lo]):
                zz = self._zz[xi + ii, yi + jj]
                rv += zz * dx * dy

        rv /= (self._dx * self._dy)

        return rv


def interp(xnew, xold, yold, left=np.nan, right=np.nan, xlog=True, ylog=True, valid=False):
    x1 = np.asarray(xnew)
    x0 = np.asarray(xold)
    y0 = np.asarray(yold)
    if xlog:
        x1 = np.log10(x1)
        x0 = np.log10(x0)
    if ylog:
        y0 = np.log10(y0)
        if (left is not None) and np.isfinite(left):
            left = np.log10(left)
        if (right is not None) and np.isfinite(right):
            right = np.log10(right)

    if valid:
        inds = (~np.isnan(x0) & ~np.isinf(x0)) & (~np.isnan(y0) & ~np.isinf(y0))
        # inds = np.where(inds)
    else:
        inds = slice(None)

    # try:
    y1 = np.interp(x1, x0[inds], y0[inds], left=left, right=right)
    # except:
    #     raise

    if ylog:
        y1 = np.power(10.0, y1)

    return y1


def interp_axis(xnew, xold, yold, axis=-1, fill=np.nan, xlog=True, ylog=True, sorted=False):
    """Linear interpolation over a particular axis.

    Arguments
    ---------
    xnew : scalar or (T,)
        New x-values to interpolate to.
    xold : ndarray (A, ...)
        Data x-coordinates.
    yold : ndarray (A, ...)
        Data y-values, must be the same shape as `xold`.

    Returns
    -------
    ynew : ndarray (..., T)
        Interpolate y-values, the same shape as `xold` and `yold`, without the `axis` dimension,
        and with a new final dimension of length (T,) for 'T' values of `xnew`.
        e.g. if `xnew` and `ynew` are (A, B, C,) and `axis=1` ===> then `ynew` will be (A, C, T)


    """
    xold = np.asarray(xold)
    yold = np.asarray(yold)
    if xold.shape != yold.shape:
        raise ValueError(f"Shapes of `xold` ({xold.shape}) and `yold` ({yold.shape}) do not match!")

    if np.isscalar(xnew):
        xnew = np.array(xnew)[tuple([np.newaxis for xx in range(xold.ndim+1)])]
    elif np.ndim(xnew) == 1:
        xnew = np.asarray(xnew)
        cut = [np.newaxis for xx in range(xold.ndim)] + [slice(None), ]
        xnew = xnew[tuple(cut)]
    else:
        raise ValueError(f"`xnew` ({np.shape(xnew)}) must be a scalar or 1D!")

    if axis != 0:
        xold = np.moveaxis(xold, axis, 0)
        yold = np.moveaxis(yold, axis, 0)

    try:
        xnew * xold[..., np.newaxis]
    except ValueError:
        raise ValueError(f"Could not broadcaast `xnew` ({xnew.shape}) with `xold` ({xold.shape})!")

    if xlog:
        xnew = np.log10(xnew)
        xold = np.log10(xold)
    if ylog:
        yold = np.log10(yold)

    if not sorted:
        idx = np.argsort(xold, axis=0)
        xold = np.take_along_axis(xold, idx, axis=0)
        yold = np.take_along_axis(yold, idx, axis=0)
        if not np.all(np.diff(xold, axis=0) >= 0.0):
            raise ValueError()

    select = (xold[..., np.newaxis] > xnew)

    aft = np.argmax(select, axis=0)
    # zero values in `aft` mean no xvals after the targets were found
    valid = (aft > 0)
    inval = ~valid
    bef = np.copy(aft)
    bef[valid] -= 1

    #   NOTE: ordering `aft` then `bef` such that `np.subtract` gives the correct sign!
    cut = np.array([aft, bef])
    cut = np.moveaxis(cut, -1, 1)
    # Get the x-values before and after the target locations  (2, N, T)
    xx = [np.take_along_axis(xold, cc, axis=0) for cc in cut]
    xx = np.moveaxis(xx, 1, -1)
    # Find how far to interpolate between values (in log-space)
    #     (N, T)
    frac = (xnew - xx[1]) / np.subtract(*xx)

    yy = [np.take_along_axis(yold, cc, axis=0) for cc in cut]
    yy = np.moveaxis(yy, 1, -1)
    # Interpolate by `frac` for each binary   (N, T) or (N, 2, T) for "double-data"
    ynew = yy[1] + (np.subtract(*yy) * frac)
    if ylog:
        ynew = np.power(10.0, ynew)

    # NOTE: this 'removes' the axis being interpolated over, so no need to `moveaxis`
    ynew = ynew[0]
    # Set invalid values to fill-value
    ynew[inval] = fill
    # Remove excess dimensions
    ynew = np.squeeze(ynew)
    return ynew


def interp_func(xold, yold, kind='mono', xlog=True, ylog=True, fill_value=np.nan, **kwargs):
    """Return an interpolation/extrapolation function constructed from the given values.

    Generally the returned function is a wrapper around `interp1d` from the `scipy.interpolate`
    module, unless `kind` is either 'mono' or 'Pchip' in which case the `PchipInterpolator` is
    used.

    """
    def in_lin(xx):
        return xx

    def in_log(xx):
        return np.log10(xx)

    def out_lin(yy):
        return yy

    def out_log(yy):
        return np.power(10.0, yy)

    if xlog:
        xold = np.log10(xold)
        in_func = in_log
    else:
        in_func = in_lin

    if ylog:
        yold = np.log10(yold)
        out_func = out_log
        if isinstance(fill_value, tuple):
            fill_value = tuple([np.log10(vv) for vv in fill_value])
        else:
            fill_value = np.log10(fill_value)

    else:
        out_func = out_lin

    if kind == 'mono' or kind.lower() == 'pchip':
        if not np.isscalar(fill_value) or not np.isnan(fill_value):
            raise ValueError("`PchipInterpolator` only support 'nan' fill values!")
        lin_interp = sp.interpolate.PchipInterpolator(
            xold, yold, extrapolate=True, **kwargs)
    else:
        lin_interp = sp.interpolate.interp1d(
            xold, yold, kind=kind, fill_value=fill_value, **kwargs)

    def ifunc(xx):
        xx = in_func(xx)
        yy = lin_interp(xx)
        yy = out_func(yy)
        return yy

    return ifunc
