"""General functions for mathematical and numerical operations.

Functions
---------
-   confidence_bands         - Bin by `xx` to calculate confidence intervals in `yy`.
-   confidence_intervals     - Compute the values bounding desired confidence intervals.
-   cumstats                 - Calculate a cumulative average and standard deviation.
-   log_normal_base_10       -
-   percentiles              -
-   stats                    - Get basic statistics for the given array.
-   stats_str                - Return a string with the statistics of the given array.
-   sigma                    - Convert from standard deviation to percentiles.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import warnings
import numpy as np

from . import math_core

__all__ = ['confidenceBands', 'confidence_bands', 'confidenceIntervals', 'confidence_intervals',
           'cumstats', 'log_normal_base_10', 'percentiles', 'sigma', 'stats', 'stats_str']


def confidenceBands(*args, **kwargs):
    """DEPRECATED: use `confidence_bands`.
    """
    warnings.warn("`confidenceBands` is deprecated.  Use `confidence_bands`",
                  DeprecationWarning, stacklevel=3)
    return confidence_bands(*args, **kwargs)


def confidence_bands(xx, yy, xbins=10, xscale='lin', confInt=[0.68, 0.95], filter=None):
    """Bin the given data with respect to `xx` and calculate confidence intervals in `yy`.

    Arguments
    ---------
    xx : array_like scalars
        Data values for the axis by which to bin.
    yy : array_like scalars
        Data values for the axis in which to calculate confidence intervals, with values
        corresponding to each of the `xx` values.  Must have the same number of elements
        as `xx`.
    xbins : int or array_like of scalar
        Specification for bins in `xx`.  Either a
        * int, describing the number of bins `N` to create automatically with scale `xscale`.
        * array_like scalar, describing the `N+1` edges of each bin (left and right).
    xscale : str
        Specification of xbin scaling if bins are to be calculated automatically, {'lin', 'log'}.
        Ignored if bin edges are given explicitly to `xbins`.
    confInt : scalar or array_like of scalar
        The percentage confidence intervals to calculate (e.g. 0.5 for median).
        Must be between {0.0, 1.0}.
    filter : str or `None`

    Returns
    -------
    (for number of bins `N`)
    count : (N,) array of int
        The number of points in each xbin.
    med : (N,) array of float
        The median value of points in each bin
    conf : array or ndarray of float
        Values describing the confidence intervals.
        If a single `confInt` is given, this will have shape (N,2);
        If `M` `confInt` values are given, this will have shape (N,M,2)
        Where in each case the 0th and 1st element of the last dimension is the lower and upper
        confidence bounds respectively.
    xbins : (N+1,) array of float
        Location of bin edges.

    """
    squeeze = False
    if not np.iterable(confInt):
        squeeze = True
        confInt = [confInt]
    xx = np.asarray(xx).flatten()
    yy = np.asarray(yy).flatten()
    if xx.shape != yy.shape:
        errStr = "Shapes of `xx` and `yy` must match ('{}' vs. '{}'."
        errStr = errStr.format(str(xx.shape), str(yy.shape))
        raise ValueError(errStr)

    # Filter based on whether `yy` values match `filter` comparison to 0.0
    if filter is not None:
        compFunc = math_core._comparison_function(filter)
        inds = compFunc(yy, 0.0)
        xx = xx[inds]
        yy = yy[inds]

    # Create bins
    xbins = math_core.asBinEdges(xbins, xx, scale=xscale)
    nbins = xbins.size - 1
    # Find the entries corresponding to each bin
    groups = math_core.groupDigitized(xx, xbins[1:], edges='right')
    # Allocate storage for results
    med = np.zeros(nbins)
    conf = np.zeros((nbins, np.size(confInt), 2))
    count = np.zeros(nbins, dtype=int)

    # Calculate medians and confidence intervals
    for ii, gg in enumerate(groups):
        count[ii] = np.size(gg)
        if count[ii] == 0: continue
        mm, cc = confidence_intervals(yy[gg], ci=confInt)
        med[ii] = mm
        conf[ii, ...] = cc[...]

    if squeeze:
        conf = conf.squeeze()

    return count, med, conf, xbins


def confidenceIntervals(*args, **kwargs):
    """DEPRECATED: use `confidence_intervals`.
    """
    warnings.warn("`confidenceIntervals` is deprecated.  Use `confidence_intervals`",
                  DeprecationWarning, stacklevel=3)
    return confidence_intervals(*args, **kwargs)


def confidence_intervals(vals, ci=None, axis=-1, filter=None, return_ci=False):
    """Compute the values bounding the target confidence intervals for an array of data.

    Arguments
    ---------
    vals : array_like of scalars
        Data over which to calculate confidence intervals.
        This can be an arbitrarily shaped ndarray.
    ci : (M,) array_like of floats
        List of desired confidence intervals as fractions (e.g. `[0.68, 0.95]`)
    axis : int or None
        Axis over which to calculate confidence intervals, or 'None' to marginalize over all axes.
    filter : str or `None`
        Filter the input array with a boolean comparison to zero.
        If no values remain after filtering, ``NaN, NaN`` is returned.

    Returns
    -------
    med : scalar
        Median of the input data.
        `None` if there are no values (e.g. after filtering).
    conf : ([L, ]M, 2) ndarray of scalar
        Bounds for each confidence interval.  Shape depends on the number of confidence intervals
        passed in `ci`, and the input shape of `vals`.
        `None` if there are no values (e.g. after filtering).
        If `vals` is 1D or `axis` is 'None', then the output shape will be (M, 2).
        If `vals` has more than one-dimension, and `axis` is not 'None', then the shape `L`
        will be the shape of `vals`, minus the `axis` axis.
        For example,
            if ``vals.shape = (4,3,5)` and `axis=1`, then `L = (4,5)`
            the final output shape will be: (4,5,M,2).

    """
    if ci is None:
        ci = [0.68, 0.95, 0.997]
    ci = np.atleast_1d(ci)
    assert np.all(ci >= 0.0) and np.all(ci <= 1.0), "Confidence intervals must be {0.0, 1.0}!"

    PERC_FUNC = np.percentile

    # Filter input values
    if filter is not None:
        # Using the filter will flatten the array, so `axis` wont work...
        kw = {}
        if (axis is not None) and np.ndim(vals) > 1:
            # err = "`filter` ({}) and `axis` ({}) arguments may be incompatible!".format(
            #     filter, axis)
            # err += "  (Shape: {})".format(np.shape(vals))
            # raise ValueError(err)
            # warnings.warn(err)
            kw['axis'] = axis

        vals = math_core.comparison_filter(vals, filter, **kw)
        vals = np.ma.filled(vals, np.nan)
        PERC_FUNC = np.nanpercentile

        if vals.size == 0:
            return np.nan, np.nan

    # Calculate confidence-intervals and median
    cdf_vals = np.array([(1.0-ci)/2.0, (1.0+ci)/2.0]).T
    # This produces an ndarray with shape `[M, 2(, L)]`
    #    If ``axis is None`` or `np.ndim(vals) == 1` then the shape will be simply `[M, 2]`
    #    Otherwise, `L` will be the shape of `vals` without axis `axis`.
    conf = [[PERC_FUNC(vals, 100.0*cdf[0], axis=axis),
             PERC_FUNC(vals, 100.0*cdf[1], axis=axis)]
            for cdf in cdf_vals]
    conf = np.array(conf)
    # Reshape from `[M, 2, L]` to `[L, M, 2]`
    if (np.ndim(vals) > 1) and (axis is not None):
        conf = np.moveaxis(conf, 2, 0)

    med = PERC_FUNC(vals, 50.0, axis=axis)
    if len(conf) == 1:
        conf = conf[0]

    if return_ci:
        return med, conf, ci

    return med, conf


def cumstats(arr):
    """Calculate a cumulative average and standard deviation.

    Arguments
    ---------
        arr <flt>[N] : input array

    Returns
    -------
        ave <flt>[N] : cumulative average over ``arr``
        std <flt>[N] : cumulative standard deviation over ``arr``

    """
    tot = len(arr)
    num = np.arange(tot)
    std = np.zeros(tot)
    # Cumulative sum
    sm1 = np.cumsum(arr)
    # Cumulative sum of squares
    sm2 = np.cumsum(np.square(arr))
    # Cumulative average
    ave = sm1/(num+1.0)

    std[1:] = np.fabs(sm2[1:] - np.square(sm1[1:])/(num[1:]+1.0))/num[1:]
    std[1:] = np.sqrt(std[1:])
    return ave, std


def sigma(sig, side='in', boundaries=False):
    """Convert from standard deviation 'sigma' to percentiles in/out-side the normal distribution.

    Arguments
    ---------
    sig : (N,) array_like scalar
        Standard deviations.
    side : str, {'in', 'out'}
        Calculate percentiles inside (i.e. [-sig, sig]) or ouside (i.e. [-inf, -sig] U [sig, inf])
    boundaries : bool
        Whether boundaries should be given ('True'), or the area ('False').

    Returns
    -------
    vals : (N,) array_like scalar
        Percentiles corresponding to the input `sig`.

    """
    import scipy as sp
    import scipy.stats

    if side.startswith('in'):
        inside = True
    elif side.startswith('out'):
        inside = False
    else:
        raise ValueError("`side` = '{}' must be {'in', 'out'}.".format(side))

    # From CDF from -inf to `sig`
    cdf = sp.stats.norm.cdf(sig)
    # Area outside of [-sig, sig]
    vals = 2.0 * (1.0 - cdf)
    # Convert to area inside [-sig, sig]
    if inside:
        vals = 1.0 - vals

    # Convert from area to locations of boundaries (fractions)
    if boundaries:
        if inside:
            vlo = 0.5*(1 - vals)
            vhi = 0.5*(1 + vals)
        else:
            vlo = 0.5*vals
            vhi = 1.0 - 0.5*vals
        return vlo, vhi

    return vals


def stats(vals, median=False):
    """Get basic statistics for the given array.

    Arguments
    ---------
        vals <flt>[N] : input array
        median <bool> : include median in return values

    Returns
    -------
        ave <flt>
        std <flt>
        [med <flt>] : median, returned if ``median`` is `True`

    """
    ave = np.average(vals)
    std = np.std(vals)
    if(median):
        med = np.median(vals)
        return ave, std, med

    return ave, std


def stats_str(data, percs=[0.0, 0.16, 0.50, 0.84, 1.00], ave=False, std=False, weights=None,
              format=None, label=None, log=False, label_log=True):
    """Return a string with the statistics of the given array.

    Arguments
    ---------
    data : ndarray of scalar
        Input data from which to calculate statistics.
    percs : array_like of scalars in {0, 100}
        Which percentiles to calculate.
    ave : bool
        Include average value in output.
    std : bool
        Include standard-deviation in output.
    format : str
        Formatting for all numerical output, (e.g. `":.2f"`).
    log : bool
        Convert values to log10 before printing.
    label_log : bool
        If `log` is also true, append a string saying these are log values.

    Output
    ------
    out : str
        Single-line string of the desired statistics.

    """
    data = np.array(data).astype(np.float)
    if log:
        data = np.log10(data)

    if format is None:
        # format = ':.2e' if log else ':.2f'
        '''
        if log:
            format = ':.2e'
        else:
            format = ':.2f'
            try:
                log_extr = np.log10(math_core.minmax(data, filter='>'))
                if np.any(log_extr >= 4) or np.any(log_extr < -2):
                    format = ':.2e'
            except AttributeError:
                pass
        '''
        format = math_core._guess_str_format_from_range(data)

    percs = np.atleast_1d(percs)
    if np.any(percs > 1.0):
        warnings.warn("WARNING: zcode.math.statistic: input `percs` should be [0.0, 1.0], "
                      "dividing these by 100.0!")
        percs /= 100.0

    percs_flag = False
    if (percs is not None) and len(percs):
        percs_flag = True

    out = ""
    # If a `format` is given, but missing the colon, add the colon
    if len(format) and not format.startswith(':'):
        format = ':' + format
    form = "{{{}}}".format(format)

    # Add average
    if ave:
        out += "ave = " + form.format(np.average(data))
        if std or percs_flag:
            out += ", "

    # Add standard-deviation
    if std:
        out += "std = " + form.format(np.std(data))
        if percs_flag:
            out += ", "

    # Add percentiles
    if percs_flag:
        tiles = percentiles(data, percs, weights=weights)
        out += "(" + ", ".join(form.format(tt) for tt in tiles) + ")"
        out += ", for (" + ", ".join("{:.0f}%".format(100*pp) for pp in percs) + ")"

    # Note if these are log-values
    if log and label_log:
        out += " (log values)"

    if label is not None:
        warnings.warn("WARNING: `label` argument is deprecated in `math_core.stats_str`",
                      stacklevel=3)
        out = label + ': ' + out

    return out


def percentiles(values, percentiles, weights=None, values_sorted=False):
    """Compute weighted percentiles.

    Copied from @Alleo answer: http://stackoverflow.com/a/29677616/230468

    Arguments
    ---------
    values: (N,)
        input data
    percentiles: (M,) scalar [0.0, 1.0]
        Desired percentiles of the data.
    weights: (N,) or `None`
        Weighted for each input data point in `values`.
    values_sorted: bool
        If True, then input values are assumed to already be sorted.

    Returns
    -------
    percs : (M,) float
        Array of percentiles of the weighted input data.

    """
    values = np.array(values).flatten()
    percentiles = np.array(percentiles, dtype=values.dtype)
    if weights is None:
        weights = np.ones_like(values)
    weights = np.array(weights)
    assert np.all(percentiles >= 0.0) and np.all(percentiles <= 1.0), \
        'percentiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        weights = weights[sorter]

    weighted_quantiles = np.cumsum(weights) - 0.5 * weights
    weighted_quantiles /= np.sum(weights)
    percs = np.interp(percentiles, weighted_quantiles, values)
    return percs


def log_normal_base_10(mu, sigma, size=None, shift=0.0):
    """Draw from a lognormal distribution with values in base-10 (instead of e).

    Arguments
    ---------
    mu : (N,) scalar
        Mean of the distribution in linear space (e.g. 1.0e8 instead of 8.0).
    sigma : (N,) scalar
        Variance of the distribution *in dex* (e.g. 1.0 means factor of 10.0 variance)
    size : (M,) int
        Desired size of sample.

    Returns
    -------
    dist : (M,...) scalar
        Resulting distribution of values (in linear space).

    """
    _sigma = np.log(10**sigma)
    dist = np.random.lognormal(np.log(mu) + shift*np.log(10.0), _sigma, size)
    return dist
