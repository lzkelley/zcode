"""General functions for mathematical and numerical operations.

Functions
---------
-   confidence_bands         - Bin by `xx` to calculate confidence intervals in `yy`.
-   confidence_intervals     - Compute the values bounding desired confidence intervals.
-   cumstats                 - Calculate a cumulative average and standard deviation.
-   stats                    - Get basic statistics for the given array.
-   stats_str                - Return a string with the statistics of the given array.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import warnings
import numpy as np

from . import math_core

__all__ = ['confidenceBands', 'confidence_bands', 'confidenceIntervals', 'confidence_intervals',
           'cumstats', 'stats', 'stats_str']


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
        compFunc = math_core._comparisonFunction(filter)
        inds = np.where(compFunc(yy, 0.0))[0]
        xx = xx[inds]
        yy = yy[inds]

    # Create bins
    xbins = math_core.asBinEdges(xbins, xx)
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
        mm, cc = confidenceIntervals(yy[gg], ci=confInt)
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


def confidence_intervals(vals, ci=[0.68, 0.95, 0.997], axis=-1, filter=None):
    """Compute the values bounding the target confidence intervals for an array of data.

    Arguments
    ---------
    vals : array_like of scalars
        Data over which to calculate confidence intervals.
    ci : (M,) array_like of floats
        List of desired confidence intervals as fractions (e.g. `[0.68, 0.95]`)
    axis : int
        Axis over which to calculate confidence intervals.
    filter : str or `None`
        Filter the input array with a boolean comparison to zero.
        If no values remain after filtering, ``None, None`` is returned.

    Returns
    -------
    med : scalar
        Median of the input data.
        `None` if there are no values (e.g. after filtering).
    conf : ndarray of scalar
        Bounds for each confidence interval.  Shape depends on the number of confidence intervals
        passed in `ci`, and also the input shape of `vals`.
        `None` if there are no values (e.g. after filtering).

    """
    ci = np.atleast_1d(ci)
    assert np.all(ci >= 0.0) and np.all(ci <= 1.0), "Confidence intervals must be {0.0, 1.0}!"

    # Filter input values
    if filter:
        vals = math_core.comparison_filter(vals, filter)
        if vals.size == 0:
            return None, None

    # Calculate confidence-intervals and median
    cdf_vals = np.array([(1.0-ci)/2.0, (1.0+ci)/2.0]).T
    conf = [[np.percentile(vals, 100.0*cdf[0], axis=axis),
             np.percentile(vals, 100.0*cdf[1], axis=axis)]
            for cdf in cdf_vals]
    conf = np.array(conf)
    med = np.percentile(vals, 50.0, axis=axis)
    if len(conf) == 1:
        conf = conf[0]

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


def stats_str(data, percs=[0, 16, 50, 84, 100], ave=True, std=False,
              format='', label='Statistics: '):
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
    label : str
        String to prepend output with, (e.g. '<label> Statistics: ...')

    Output
    ------
    out : str
        Single-line string of the desired statistics.

    """
    data = np.asarray(data)
    percs = np.atleast_1d(percs)
    percs_flag = False
    if percs is not None and len(percs): percs_flag = True

    out = label
    form = "{{{}}}".format(format)
    if ave:
        out += "ave = " + form.format(np.average(data))
        if std or percs_flag:
            out += ", "
    if std:
        out += "std = " + form.format(np.std(data))
        if percs_flag:
            out += ", "
    if percs_flag:
        tiles = np.percentile(data, percs)
        out += "percentiles: [" + ", ".join(form.format(tt) for tt in tiles) + "]"
        out += ", for (" + ", ".join("{:.1f}%".format(pp) for pp in percs) + ")"

    return out
