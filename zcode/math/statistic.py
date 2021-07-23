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
import scipy as sp
import scipy.stats  # noqa

from zcode import utils
from zcode.math import math_core


__all__ = [
    'confidence_bands', 'confidence_intervals',
    'cumstats', 'frac_str', 'info', 'log_normal_base_10', 'mean',
    'percs_from_sigma', 'quantiles', 'random_power', 'sigma',
    'stats', 'stats_str', 'std',
    'LH_Sampler',
    # DEPRECATED
    'percentiles'
]


def confidence_bands(xx, yy, xbins=10, xscale='lin', percs=[0.68, 0.95], filter=None):
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
    if not np.iterable(percs):
        squeeze = True
        percs = [percs]
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
    conf = np.zeros((nbins, np.size(percs), 2))
    count = np.zeros(nbins, dtype=int)

    # Calculate medians and confidence intervals
    for ii, gg in enumerate(groups):
        count[ii] = np.size(gg)
        if count[ii] == 0: continue
        mm, cc = confidence_intervals(yy[gg], percs=percs)
        med[ii] = mm
        conf[ii, ...] = cc[...]

    if squeeze:
        conf = conf.squeeze()

    return count, med, conf, xbins


def confidence_intervals(vals, sigma=None, percs=None, weights=None, axis=None,
                         filter=None, return_ci=False,
                         # DEPRECATED ARGUMENTS:
                         ci=None):
    """Compute the values bounding the target confidence intervals for an array of data.

    Arguments
    ---------
    vals : array_like of scalars
        Data over which to calculate confidence intervals.
        This can be an arbitrarily shaped ndarray.
    sigma : (M,) array_like of float
        Confidence values as standard-deviations, converted to percentiles.
    percs : (M,) array_like of floats
        List of desired confidence intervals as fractions (e.g. `[0.68, 0.95]`)
    axis : int or None
        Axis over which to calculate confidence intervals, or 'None' to marginalize over all axes.
    filter : str or `None`
        Filter the input array with a boolean comparison to zero.
        If no values remain after filtering, ``NaN, NaN`` is returned.
    return_ci : bool
        Return the confidence-interval values used (i.e. percentiles)
    ci : DEPRECATED, use `percs` instead

    Returns
    -------
    med : scalar
        Median of the input data.
        `None` if there are no values (e.g. after filtering).
    conf : ([L, ]M, 2) ndarray of scalar
        Bounds for each confidence interval.  Shape depends on the number of confidence intervals
        passed in `percs`, and the input shape of `vals`.
        `None` if there are no values (e.g. after filtering).
        If `vals` is 1D or `axis` is 'None', then the output shape will be (M, 2).
        If `vals` has more than one-dimension, and `axis` is not 'None', then the shape `L`
        will be the shape of `vals`, minus the `axis` axis.
        For example,
            if ``vals.shape = (4,3,5)` and `axis=1`, then `L = (4,5)`
            the final output shape will be: (4,5,M,2).
    percs : (M,) ndarray of float, optional
        The percentile-values used for calculating confidence intervals.
        Only returned if `return_ci` is True.

    """
    percs = utils.dep_warn_var("ci", ci, "percs", percs)

    if percs is not None and sigma is not None:
        raise ValueError("Only provide *either* `percs` or `sigma`!")

    if percs is None:
        if sigma is None:
            sigma = [1.0, 2.0, 3.0]
        percs = percs_from_sigma(sigma)

    percs = np.atleast_1d(percs)
    if np.any(percs < 0.0) or np.all(percs > 1.0):
        raise ValueError("`percs` must be [0.0, 1.0]!   {}".format(stats_str(percs)))

    # PERC_FUNC = np.percentile
    def PERC_FUNC(xx, pp, **kwargs):
        return quantiles(xx, pp/100.0, weights=weights, **kwargs)

    # Filter input values
    if filter is not None:
        # Using the filter will flatten the array, so `axis` wont work...
        kw = {}
        if (axis is not None) and np.ndim(vals) > 1:
            kw['axis'] = axis

        if weights is not None:
            raise NotImplementedError("`weights` argument does not work with `filter`!")

        vals = math_core.comparison_filter(vals, filter, mask=True)  # , **kw)
        # vals = np.ma.filled(vals, np.nan)
        # PERC_FUNC = np.nanpercentile  # noqa

        if vals.size == 0:
            return np.nan, np.nan

    # Calculate confidence-intervals and median
    cdf_vals = np.array([(1.0-percs)/2.0, (1.0+percs)/2.0]).T
    # This produces an ndarray with shape `[M, 2(, L)]`
    #    If ``axis is None`` or `np.ndim(vals) == 1` then the shape will be simply `[M, 2]`
    #    Otherwise, `L` will be the shape of `vals` without axis `axis`.
    conf = [[PERC_FUNC(vals, 100.0*cdf[0], axis=axis),
             PERC_FUNC(vals, 100.0*cdf[1], axis=axis)]
            for cdf in cdf_vals]
    conf = np.array(conf)
    # Reshape from `[M, 2, L]` to `[L, M, 2]`
    if (np.ndim(vals) > 1) and (axis is not None):
        conf = np.moveaxis(conf, -1, 0)

    med = PERC_FUNC(vals, 50.0, axis=axis)
    if len(conf) == 1:
        conf = conf[0]

    if return_ci:
        return med, conf, percs

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


def frac_str(num, den=None, frac_fmt=None, dec_fmt=None):
    """Create a string of the form '{}/{} = {}' for reporting fractional values.
    """
    if den is None:
        assert num.dtype == bool, "If no `den` is given, array must be boolean!"
        den = num.size
        num = np.count_nonzero(num)

    try:
        dec_frac = num / den
    except ZeroDivisionError:
        dec_frac = np.nan

    if frac_fmt is None:
        frac_exp = np.fabs(np.log10([num, den]))

        if np.any(frac_exp >= 4):
            frac_fmt = ".1e"
        else:
            frac_fmt = "d"

    if dec_fmt is None:
        dec_exp = np.fabs(np.log10(dec_frac))
        if dec_exp > 3:
            dec_fmt = ".3e"
        else:
            dec_fmt = ".4f"

    fstr = "{num:{ff}}/{den:{ff}} = {frac:{df}}".format(
        num=num, den=den, frac=dec_frac, ff=frac_fmt, df=dec_fmt)

    return fstr


def info(array, shape=True, sample=3, stats=True):
    rv = ""
    if shape:
        rv += "{} ".format(np.shape(array))
    if (sample is not None) and (sample > 0):
        rv += "{} ".format(math_core.str_array(array, sides=sample))
    if stats:
        rv += "{} ".format(stats_str(array, label=False))

    return rv


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


def mean(vals, weights=None, **kwargs):
    if weights is None:
        return np.mean(vals, **kwargs)

    ave = np.sum(vals*weights, **kwargs) / np.sum(weights, **kwargs)
    return ave


def percentiles(*args, **kwargs):
    utils.dep_warn("percentiles", newname="quantiles")
    return quantiles(*args, **kwargs)


def quantiles(values, percs=None, sigmas=None, weights=None, axis=None,
              values_sorted=False, filter=None):
    """Compute weighted percentiles.

    Copied from @Alleo answer: http://stackoverflow.com/a/29677616/230468
    NOTE: if `values` is a masked array, then only unmasked values are used!

    Arguments
    ---------
    values: (N,)
        input data
    percs: (M,) scalar [0.0, 1.0]
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
    if filter is not None:
        values = math_core.comparison_filter(values, filter)

    if not isinstance(values, np.ma.MaskedArray):
        values = np.asarray(values)

    if percs is None:
        percs = sp.stats.norm.cdf(sigmas)

    if np.ndim(values) > 1:
        if axis is None:
            values = values.flatten()
    else:
        if axis is not None:
            raise ValueError("Cannot act along axis '{}' for 1D data!".format(axis))

    percs = np.array(percs)
    if weights is None:
        weights = np.ones_like(values)
    weights = np.array(weights)
    try:
        weights = np.ma.masked_array(weights, mask=values.mask)
    except AttributeError:
        pass

    assert np.all(percs >= 0.0) and np.all(percs <= 1.0), 'percentiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values, axis=axis)
        values = np.take_along_axis(values, sorter, axis=axis)
        weights = np.take_along_axis(weights, sorter, axis=axis)

    if axis is None:
        weighted_quantiles = np.cumsum(weights) - 0.5 * weights
        weighted_quantiles /= np.sum(weights)
        percs = np.interp(percs, weighted_quantiles, values)
        return percs

    weights = np.moveaxis(weights, axis, -1)
    values = np.moveaxis(values, axis, -1)

    weighted_quantiles = np.cumsum(weights, axis=-1) - 0.5 * weights
    weighted_quantiles /= np.sum(weights, axis=-1)[..., np.newaxis]
    # weighted_quantiles = np.moveaxis(weighted_quantiles, axis, -1)
    percs = [np.interp(percs, weighted_quantiles[idx], values[idx])
             for idx in np.ndindex(values.shape[:-1])]
    percs = np.array(percs)
    return percs


def percs_from_sigma(sigma, side='in', boundaries=False):
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
    if side.startswith('in'):
        inside = True
    elif side.startswith('out'):
        inside = False
    else:
        raise ValueError("`side` = '{}' must be {'in', 'out'}.".format(side))

    # From CDF from -inf to `sig`
    cdf = sp.stats.norm.cdf(sigma)
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


def random_power(extr, pdf_index, size=1, **kwargs):
    """Draw from power-law PDF with the given extrema and index.

    FIX/BUG : negative `extr` values break `pdf_index=-1` !!

    Arguments
    ---------
    extr : array_like scalar
        The minimum and maximum value of this array are used as extrema.
    pdf_index : scalar
        The power-law index of the PDF distribution to be drawn from.  Any real number is valid,
        positive or negative.
        NOTE: the `numpy.random.power` function uses the power-law index of the CDF, i.e. `g+1`
    size : scalar
        The number of points to draw (cast to int).
    **kwags : dict pairs
        Additional arguments passed to `zcode.math_core.minmax` with `extr`.

    Returns
    -------
    rv : (N,) scalar
        Array of random variables with N=`size` (default, size=1).

    """
    # if not np.isscalar(pdf_index):
    #     err = "`pdf_index` (shape {}; {}) must be a scalar value!".format(
    #         np.shape(pdf_index), pdf_index)
    #     raise ValueError(err)

    extr = math_core.minmax(extr, **kwargs)
    if pdf_index == -1:
        rv = 10**np.random.uniform(*np.log10(extr), size=int(size))
    else:
        rr = np.random.random(size=int(size))
        gex = extr ** (pdf_index+1)
        rv = (gex[0] + (gex[1] - gex[0])*rr) ** (1./(pdf_index+1))

    return rv


def sigma(*args, **kwargs):
    # ---- DECPRECATION SECTION ----
    utils.dep_warn("sigma", newname="percs_from_sigma")
    # ------------------------------
    return percs_from_sigma(*args, **kwargs)


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
              format=None, log=False, label=True, label_log=True, filter=None):
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
    label : bool
        Add label for which percentiles are being printed
    label_log : bool
        If `log` is also true, append a string saying these are log values.

    Output
    ------
    out : str
        Single-line string of the desired statistics.

    """
    # data = np.array(data).astype(np.float)
    data = np.array(data)
    if filter is not None:
        data = math_core.comparison_filter(data, filter)
        if np.size(data) == 0:
            return "empty after filtering"

    if log:
        data = np.log10(data)

    percs = np.atleast_1d(percs)
    if np.any(percs > 1.0):
        warnings.warn("WARNING: zcode.math.statistic: input `percs` should be [0.0, 1.0], "
                      "dividing these by 100.0!")
        percs /= 100.0

    percs_flag = False
    if (percs is not None) and len(percs):
        percs_flag = True

    out = ""

    if format is None:
        allow_int = False if (ave or std) else True
        format = math_core._guess_str_format_from_range(data, allow_int=allow_int)

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
        tiles = quantiles(data, percs, weights=weights).astype(data.dtype)
        out += "(" + ", ".join(form.format(tt) for tt in tiles) + ")"
        if label:
            out += ", for (" + ", ".join("{:.0f}%".format(100*pp) for pp in percs) + ")"

    # Note if these are log-values
    if log and label_log:
        out += " (log values)"

    return out


def std(vals, weights=None, **kwargs):
    """

    See: https://www.itl.nist.gov/div898/software/dataplot/refman2/ch2/weightsd.pdf
    """
    if weights is None:
        return np.std(vals, **kwargs)

    mm = np.count_nonzero(weights)
    ave = mean(vals, weights=weights, **kwargs)
    num = np.sum(weights * (vals - ave)**2)
    den = np.sum(weights) * (mm - 1) / mm
    std = np.sqrt(num/den)
    return std


class LH_Sampler:
    """

    Much of this code was taken from the pyDOE project:
        - https://github.com/tisimst/pyDOE

    This code was originally published by the following individuals for use with
    Scilab:
        Copyright (C) 2012 - 2013 - Michael Baudin
        Copyright (C) 2012 - Maria Christopoulou
        Copyright (C) 2010 - 2011 - INRIA - Michael Baudin
        Copyright (C) 2009 - Yann Collette
        Copyright (C) 2009 - CEA - Jean-Marc Martinez

        website: forge.scilab.org/index.php/p/scidoe/sourcetree/master/macros
    Much thanks goes to these individuals. It has been converted to Python by
    Abraham Lee.
    """

    '''
    @classmethod
    def oversample(cls, npar, nsamp, oversamp, **kwargs):
        if not isinstance(oversamp, int) or oversamp < 1:
            raise ValueError(f"`oversamp` argument '{oversamp}' must be an integer!")

        samples = None
        for ii in range(oversamp):
            ss = cls.sample(npar, nsamp=nsamp, **kwargs)
            if samples is None:
                samples = ss
            else:
                samples = np.append(samples, ss, axis=-1)

        return samples
    '''

    @classmethod
    def sample(cls, vals, nsamp=None, **kwargs):
        if isinstance(vals, int):
            return cls.sample_unit(vals, nsamp=nsamp, **kwargs)

        return cls.sample_vals(vals, nsamp=nsamp, **kwargs)

    @classmethod
    def sample_vals(cls, vals, nsamp=None, log=False, **kwargs):
        vals = np.asarray(vals)
        try:
            npar, check = np.shape(vals)
            if (check != 2) or (npar < 2):
                raise ValueError
        except ValueError:
            print(f"vals = {vals}")
            raise ValueError(f"Shape of `vals` ({np.shape(vals)}) must be (N,2)!")

        if np.isscalar(log):
            log = [log] * npar

        if np.any([ll not in [True, False] for ll in log]):
            raise ValueError(f"`log` value(s) must be 'True' or 'False'!")

        # Draw samples in [0.0, 1.0]
        samps = cls.sample_unit(npar, nsamp=nsamp, **kwargs)
        # Map samples to the given ranges in log or linear space
        for ii, vv in enumerate(vals):
            if log[ii]:
                vv = np.log10(vv)

            # temp = np.copy(samps[ii, :])
            # samps[ii, :] *= (vv.max() - vv.min())
            # samps[ii, :] += vv.min()
            samps[ii, :] = (vv.max() - vv.min()) * samps[ii, :] + vv.min()

            if log[ii]:
                samps[ii, :] = 10.0 ** samps[ii, :]
                vv = 10.0 ** vv

            # if np.any((samps[ii] < vv.min()) | (samps[ii] > vv.max())):
            #     print(f"temp = {temp}")
            #     print(f"vv = {vv}")
            #     err = (
            #         f"Samples ({stats_str(samps[ii])}) exceeded "
            #         f"values ({math_core.minmax(vv)})"
            #     )
            #     raise ValueError(err)

        return samps

    @classmethod
    def sample_unit(cls, npar, nsamp=None, center=False, optimize=None, iterations=10):
        if nsamp is None:
            nsamp = npar

        # Construct optimization variables/functions
        optimize = None if (optimize is None) else optimize.lower()
        if optimize is not None:
            if optimize.startswith('dist'):
                extr = 0.0
                mask = np.ones((nsamp, nsamp), dtype=bool)
                comp = np.less

                # Minimum euclidean distance between points
                def metric(xx):
                    dist = (xx[:, np.newaxis, :] - xx[:, :, np.newaxis])**2
                    dist = np.sum(dist, axis=0)
                    return np.min(dist[mask])

            elif optimize.startswith('corr'):
                extr = np.inf
                mask = np.ones((npar, npar), dtype=bool)
                comp = np.greater

                # Maximum correlation
                metric = lambda xx: np.max(np.abs(np.corrcoef(xx)[mask]))

            np.fill_diagonal(mask, False)

        # iterate over randomizations
        for ii in range(iterations):
            cand = cls._sample(npar, nsamp, center=center)
            if optimize is None:
                samples = cand
                break

            # -- Optimize
            # Calculate the metric being optimized
            met = metric(cand)
            # Compare the metric to the previous extrema and store new values if better
            if comp(extr, met):
                extr = met
                samples = cand

        return samples

    @classmethod
    def _sample(cls, npar, nsamp, center=False):
        # Generate the intervals
        cut = np.linspace(0, 1, nsamp + 1)
        lo = cut[:-1]
        hi = cut[1:]

        # Fill points uniformly in each interval
        shape = (npar, nsamp)  # , nreals)
        if center:
            points = np.zeros(shape)
            points[...] = 0.5 * (lo + hi)[np.newaxis, :]
        else:
            points = np.random.uniform(size=shape)
            points = points * (hi - lo)[np.newaxis, :] + lo[np.newaxis, :]

        for j in range(npar):
            points[j, :] = np.random.permutation(points[j, :])

        return points
