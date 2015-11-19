"""General functions for mathematical and numerical operations.

Functions
---------
-   spline                   - Create a general spline interpolation function.
-   contiguousInds           - Find the largest segment of contiguous array values
-   cumtrapz_loglog          - Perform a cumulative integral in log-log space.
-   within                   - Test whether a value is within the bounds of another.
-   indsWithin               - Find the indices within the given extrema.
-   minmax                   - Find the min and max of given values.
-   really1d                 - Test whether an array_like is really 1D (e.g. not jagged array).
-   argextrema               - Find the index of the extrema in the input array.
-   spacing                  - Create an even spacing between extrema from given data.
-   asBinEdges               - Create bin-edges if the given `bins` do not already give them.
-   strArray                 - Create a string representation of a numerical array.
-   sliceForAxis             - Array slicing object which slices only the target axis.
-   midpoints                - Return the midpoints between values in the given array.
-   vecmag                   - find the magnitude/distance of/between vectors.
-   extend                   - Extend the given array by extraplation.
-   renumerate               - construct a reverse enumeration iterator.
-   cumstats                 - Calculate a cumulative average and standard deviation.
-   confidenceIntervals      - Compute the values bounding desired confidence intervals.
-   confidenceBands          - Bin by `xx` to calculate confidence intervals in `yy`.
-   frexp10                  - Decompose a float into mantissa and exponent (base 10).
-   stats                    - Get basic statistics for the given array.
-   groupDigitized           - Get a list of array indices corresponding to each bin.
-   sampleInverse            - Find x-sampling to evenly divide a function in y-space.
-   smooth                   - Use convolution to smooth the given array.
-   mono                     - Check for monotonicity in the given array.

-   _trapezium_loglog
-   _comparisonFunction
-   _fracToInt

"""
from __future__ import absolute_import, division, print_function, unicode_literals

from six.moves import xrange

import numpy as np
import scipy as sp
import scipy.interpolate
import warnings
import numbers

__all__ = ['spline', 'contiguousInds', 'cumtrapz_loglog',
           'within', 'indsWithin', 'minmax', 'really1d',
           'argextrema', 'spacing', 'asBinEdges',
           'strArray', 'sliceForAxis', 'midpoints',
           'vecmag', 'extend',
           'renumerate', 'cumstats', 'confidenceIntervals', 'confidenceBands',
           'frexp10', 'stats', 'groupDigitized',
           'sampleInverse', 'smooth', 'mono']


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

    xp = np.array(xx)
    yp = np.array(yy)

    # Make sure arguments are sorted (by independent variable `xx`)
    if(sort):
        inds = np.argsort(xp)
        xp = xp[inds]
        yp = yp[inds]

    # Select positive y-values
    if(pos):
        inds = np.where(yp > 0.0)[0]
        xp = xp[inds]
        yp = yp[inds]

    # Convert to log-space as needed
    if(log):
        xp = np.log10(xp)
        yp = np.log10(yp)

    # Sort input arrays
    inds = np.argsort(xp)
    xp = xp[inds]
    yp = yp[inds]

    # Monotonic Interpolation
    if(mono):
        if(order != 3): warnings.warn("monotonic `PchipInterpolator` is always cubic!")
        terp = sp.interpolate.PchipInterpolator(xp, yp, extrapolate=extrap)
    # General Interpolation
    else:
        # Let function extrapolate outside range
        if(extrap): ext = 0
        # Return zero outside of range
        else:       ext = 1
        terp = sp.interpolate.InterpolatedUnivariateSpline(xp, yp, k=order, ext=ext)

    # Convert back to normal space, as needed
    if(log): spline = lambda xx, terp=terp: np.power(10.0, terp(np.log10(xx)))
    else:    spline = terp

    return spline


def contiguousInds(args):
    """Find the longest contiguous segment of positive values in the array.
    """
    condition = (np.array(args) > 0.0)

    # Find the indicies of changes in ``condition``
    dd = np.diff(condition)
    idx, = dd.nonzero()

    # Start things after change in ``condition``, thus shift indices 1 rightward
    idx += 1

    # If the start is True prepend a 0
    if condition[0]:  idx = np.r_[0, idx]

    # If the end is True, append the length of the array
    if condition[-1]: idx = np.r_[idx, condition.size]

    # Reshape the result into two columns
    idx.shape = (-1, 2)

    # Find lengths of each contiguous segment
    sizes = np.diff(idx, axis=1)
    # Find the location of maximum segment length
    maxPos = np.argmax(sizes)
    # Make indices spanning longest segment
    inds = np.arange(*idx[maxPos])

    return inds


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


def within(vals, extr, edges=True, all=False, inv=False):
    """Test whether a value or array is within the bounds of another.

    Arguments
    ---------
       vals   <scalar>([N]) : test value(s)
       extr  <scalar>[M]    : array or span to compare with
       edges <bool>         : optional, include the boundaries of ``extr`` as 'within'
       all   <bool>         : optional, whether All values are within bounds (single return `bool`)
       inv   <bool>         : optional, invert results (i.e. `True` ==> `False` and visa-versa)

    Returns
    -------
       <bool> : True if within, False otherwise

    """

    extr_bnds = minmax(extr)

    # Include edges for WITHIN bounds (thus not including is outside)
    if(edges): retval = np.asarray(((vals >= extr_bnds[0]) & (vals <= extr_bnds[1])))
    # Don't include edges for WITHIN  (thus include them for outside)
    else:      retval = np.asarray(((vals > extr_bnds[0]) & (vals < extr_bnds[1])))

    # Convert to single return value
    if(all): retval = np.all(retval)

    # Invert results
    if(inv): retval = np.invert(retval)

    return retval


def indsWithin(vals, extr, edges=True):
    """Find the indices of the input array which are within the given extrema.
    """
    assert np.ndim(vals) == 1, "Only `ndim = 1` arrays allowed!"
    bnds = minmax(extr)
    if(edges):
        inds = np.where((vals >= bnds[0]) & (vals <= bnds[1]))[0]
    else:
        inds = np.where((vals > bnds[0]) & (vals < bnds[1]))[0]

    return inds


def minmax(data, nonzero=False, positive=False, prev=None, stretch=0.0):
    """Find minimum and maximum of given data, return as numpy array.

    If ``prev`` is provided, the returned minmax values will also be compared to it.

    Arguments
    ---------
       data     <scalar>[...] : arbitrarily shaped data to find minumum and maximum of.
       nonzero  <bool>        : ignore zero values in input ``data``
       positive <bool>        : select values '>= 0.0' in input ``data``
       prev     <scalar>[2]   : also find min/max against prev[0] and prev[1] respectively.
       stretch  <flt>         : factor by which to stretch min and max by (1 +- ``stretch``)

    Returns
    -------
       minmax <scalar>[2] : minimum and maximum of given data (and ``prev`` if provided).
                            Returned data type will match the input ``data`` (and ``prev``).

    To-Do
    -----
     - Added an 'axis' argument, remove 'flatten()' to accomodate arbitrary shapes

    """

    useData = np.array(data).flatten()

    # Filter out zeros if desired
    if(nonzero):  useData = np.array(useData[np.nonzero(useData)])
    if(positive): useData = np.array(useData[np.where(useData >= 0.0)])

    # If there are no elements (left), return ``prev`` (`None` if not provided)
    if(np.size(useData) == 0): return prev

    # Determine stretch factor
    lef = (1.0-stretch)
    rit = (1.0+stretch)

    # Find extrema
    minmax = np.array([lef*np.min(useData), rit*np.max(useData)])

    # Compare to previous extrema, if given
    if(prev is not None):
        minmax[0] = np.min([minmax[0], prev[0]])
        minmax[1] = np.max([minmax[1], prev[1]])

    return minmax


def really1d(arr):
    """Test whether an array_like is really 1D (i.e. not a jagged ND array).

    Test whether the input array is uniformly one-dimensional, as apposed to (e.g.) a ``ndim == 1``
    list or array of irregularly shaped sub-lists/sub-arrays.  True for an empty list `[]`.

    Arguments
    ---------
    arr : array_like
        Array to be tested.

    Returns
    -------
    bool
        Whether `arr` is purely 1D.

    """
    if np.ndim(arr) != 1:
        return False
    # Empty list or array
    if len(arr) == 0:
        return True
    if np.any(np.vectorize(np.ndim)(arr)):
        return False
    return True


def argextrema(arr, type, filter=None):
    """Find the index of the desired type of extrema in the input array.
    """
    # Valid filters, NOTE: do *not* include 'e' (equals), doesn't make sense here.
    good_filter = [None, 'g', 'ge', 'l', 'le']
    if(filter not in good_filter):
        raise ValueError("Filter '%s' Unrecognized." % (type))
    # Make sure `type` is valid
    good_type = ['min', 'max']
    if(not np.any([type.startswith(gt) for gt in good_type])):
        raise ValueError("Type '%s' Unrecognized." % (type))
    # Make sure input array `arr` is valid (1D)
    arr = np.asarray(arr)
    if(arr.ndim != 1 or arr[0].ndim != 0):
        raise ValueError("Only 1D arrays currently supported.")

    if(type.startswith('min')):
        func = np.argmin
    elif(type.startswith('max')):
        func = np.argmax

    # Find whether the `filter` criteria is True
    if(filter):
        filterFunc = _comparisonFunction(filter)
        sel = filterFunc(arr, 0.0)
    # If no filter (`None`), all values are valid
    else:
        sel = np.ones_like(arr, dtype=bool)

    # Find the extrema within the valid (`sel`) subset
    ind = func(arr[sel])
    # Convert to index wrt the full input array
    ind = np.where(sel)[0][ind]
    return ind


def spacing(data, scale='log', num=100, nonzero=None, positive=None):
    """Create an evenly spaced array between extrema from the given data.

    If ``nonzero`` and ``positive`` are not given, educated guesses are made based on ``scale``.

    Arguments
    ---------
       data     <scalar>[M] : data from which to extract the extrema for bounds
       scale    <str>       : optional, scaling for spacing, {'lin', 'log'}
       num      <int>       : optional, number of points, ``N``
       nonzero  <bool>      : optional, only use '!= 0.0' elements of ``data``
       positive <bool>      : optional, only use '>= 0.0' elements of ``data``

    Returns
    -------
       spacing <scalar>[N] : array of evenly spaced points, with number of elements ``N = num``

    """

    if(scale.startswith('log')): log_flag = True
    elif(scale.startswith('lin')): log_flag = False
    else: raise RuntimeError("``scale`` '%s' unrecognized!" % (scale))

    if(nonzero is None):
        if(log_flag): nonzero = True
        else:         nonzero = False

    if(positive is None):
        if(log_flag): positive = True
        else:         positive = False

    span = minmax(data, nonzero=nonzero, positive=positive)
    if(log_flag): spacing = np.logspace(*np.log10(span), num=num)
    else:         spacing = np.linspace(*span,           num=num)

    return spacing


def asBinEdges(bins, data, scale='lin'):
    """Create bin-edges if the given `bins` do not already give them.

    Code based on 'scipy.stats._binned_statistic.binned_statistic_dd'.

    Arguments
    ---------
    bins : int, array_like of int, or array_like of scalars
        Specification for bins.  This must be one of:
        * int, specifying the number of bins to create with scaling `scale`, in *all* dimensions.
        * array_like of int, specifying the number of bins to create in *each* dimension.
        * array_like of scalars, specifying the (left and right) edges for bins of *1D* data.
        * list of array_like of scalars, specifying the (left and right) edges for bins in
          each dimension.
    data : (M,[D]) array_like of scalars
        Input data with which to construct bin-edges, or determine validity of given bin-edges.
        This can be one-dimensional with shape (M), or D-dimensional with shape (M,D).
    scale : str
        Specification for spacing of bin edges {'lin', 'log'}.  Ignored if bin-edges are given
        explicitly in `bins`.

    Returns
    -------
    bins : list of array_like float
        For 1D input data, this will be a single array of `N+1` bin edges for `N` bins.
        If the input data is `D`-Dimensional, then this will be a list of `D` arrays of bin edges,
        each element of which can have a different number of bins.

    """
    data = np.asarray(data)
    ndim = data.ndim
    edges = ndim * [None]

    try:
        M = len(bins)
        if M != ndim:
            flag_1d = really1d(bins)
            if flag_1d:
                bins = [np.asarray(bins, float)]

            if not flag_1d or len(bins) != ndim:
                raise ValueError('The dimension of bins must be equal '
                                 'to the dimension of the sample x.')
    except TypeError:
        bins = ndim * [bins]

    # If `scale` is a single value for all dimensions
    if(np.ndim(scale) == 0):
        scale = ndim * [scale]
    # otherwise `scale` must be specified for each dimension
    elif(np.size(scale) != ndim):
        raise ValueError("`scale` must be a single value or a value for each dimension.")

    # Find range for each dimension
    smin = np.atleast_1d(np.array(data.min(axis=0), float))
    smax = np.atleast_1d(np.array(data.max(axis=0), float))
    # Make sure the bins have a finite width.
    for i in xrange(len(smin)):
        if smin[i] == smax[i]:
            smin[i] = smin[i] - 0.5
            smax[i] = smax[i] + 0.5

    # Create arrays describing edges of bins
    for ii in xrange(ndim):
        if np.isscalar(bins[ii]):
            edges[ii] = spacing([smin[ii], smax[ii]], scale[ii], num=bins[ii] + 1)
        else:
            edges[ii] = np.asarray(bins[ii], float)

    if(ndim == 1):
        edges = edges[0]

    return edges


def strArray(arr, first=4, last=4, delim=", ", format=".4e"):
    """Create a string representation of a numerical array.

    Arguments
    ---------
    arr : array_like scalars,
        Array to be converted to string.
    first : int or None,
        Number of elements at the beginning of the array to print.
        `None` means FULL array, while `0` means zero elements.
    last : int,
        Number of elements at the end of the array to print.
    delim : str,
        Character to delimit elements of string array.
    format : str,
        Specification of how each array element should be converted to a str.
        This is a c-style specification used by ``str.format``.

    Returns
    -------
    arrStr : str,
        Stringified version of input array.

    """

    if(first is None or last is None):
        first = None
        last = 0

    # Create the style specification
    form = "{:%s}" % (format)

    arrStr = "["
    # Add the first `first` elements
    if(first or first is None):
        arrStr += delim.join([form.format(vv) for vv in arr[:first]])

    # Include separator unless full array is being printed
    if(first is not None): arrStr += "... "

    # Add the last `last` elements
    if(last):
        arrStr += delim.join([form.format(vv) for vv in arr[-last-1:]])

    arrStr += "]"

    return arrStr


def sliceForAxis(arr, axis=-1, start=None, stop=None, step=None):
    """
    Creates an array slicing object which slices only the target axis.

    If ``arr`` is a single number, it is taken as the number of dimensions to create the slice for.
    Otherwise, the ndim of ``arr`` is used.

    Arguments
    ---------
        arr   <obj>    : integer number of dimensions, or N-Dim array of objects to retrieve ndim
        axis  <int>    : target axis (`-1` for last)
        start <int>    : None for default
        stop  <int>    : None for default
        step  <int>    : None for default

    Returns
    -------
        cut   <obj>[N] : list of `slice` objects for each dimension, only slicing on ``axis``

    """

    if(start is stop is step is None):
        raise RuntimeError("``start``,``stop``, or ``step`` required!")

    ndim = np.ndim(arr)
    if(ndim == 0): ndim = arr

    if(ndim > 1):
        #     Create an object to slice all elements of all dims
        cut = [slice(None)]*ndim
        #     Exclude the last element of the last dimension
        cut[axis] = slice(start, stop, step)
    else:
        if(axis != 0 and axis != -1): raise RuntimeError("cannot slice nonexistent axis!")
        cut = slice(start, stop, step)

    return cut


def midpoints(arr, log=False, frac=0.5, axis=-1, squeeze=True):
    """Return the midpoints between values in the given array.

    If the given array is N-dimensional, midpoints are calculated from the last dimension.

    Arguments
    ---------
        arr : ndarray of scalars,
            Input array.
        log : bool,
            Find midpoints in log-space.
        frac : float,
            Fraction of the way between intervals (e.g. `0.5` for half-way midpoints).
        axis : int,
            Which axis about which to find the midpoints.

    Returns
    -------
        mids : ndarray of floats,
            The midpoints of the input array.
            The resulting shape will be the same as the input array `arr`, except that
            `mids.shape[axis] == arr.shape[axis]-1`.

    """

    if(np.shape(arr)[axis] < 2):
        raise RuntimeError("Input ``arr`` does not have a valid shape!")

    # Convert to log-space
    if(log): user = np.log10(arr)
    else:    user = np.array(arr)

    diff = np.diff(user, axis=axis)

    #     skip the last element, or the last axis
    cut = sliceForAxis(user, axis=axis, stop=-1)
    start = user[cut]
    mids = start + frac*diff

    if(log): mids = np.power(10.0, mids)
    if(squeeze): mids = mids.squeeze()

    return mids


def vecmag(r1, r2=None):
    """Calculate the distance from vector(s) r1 to r2.

    Both ``r1`` and ``r2`` can be either single, ``M`` dimensional, vectors or a set of ``N`` of
    them.  If both ``r1`` and ``r2`` are sets of vectors, they must have the same length.

    Arguments
    ---------
       r1 <scalar>[(N,)M] : first  vector (set)
       r2 <scalar>[(N,)M] : second vector (set)

    Returns
    -------
       dist <scalar>([N]) : distances

    """

    if(r2 is None): r2 = np.zeros(np.shape(r1))

    if(len(np.shape(r1)) > 1 or len(np.shape(r2)) > 1):
        dist = np.sqrt(np.sum(np.square(r1 - r2), axis=1))
    else:
        dist = np.sqrt(np.sum(np.square(r1 - r2)))

    return dist


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


def renumerate(arr):
    """
    Same as ``enumerate`` but in reverse order.  Uses iterators, no copies made.
    """
    return zip(reversed(range(len(arr))), reversed(arr))


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


def confidenceIntervals(vals, ci=[0.68, 0.95, 0.997], axis=-1):
    """Compute the values bounding the target confidence intervals for an array of data.

    Arguments
    ---------
       vals <scalar>[N] : array of sample data points
       ci   <scalar>[M] : optional, list of confidence intervals as fractions (e.g. `[0.68, 0.95]`)

    Returns
    -------
       med  <scalar>      : median of the data
       conf <scalar>[M,2] : bounds for each confidence interval

    """

    if(not np.iterable(ci)): ci = [ci]
    ci = np.asarray(ci)
    assert np.all(ci >= 0.0) and np.all(ci <= 1.0), "Confidence intervals must be {0.0,1.0}!"

    cdf_vals = np.array([(1.0-ci)/2.0, (1.0+ci)/2.0]).T
    conf = [[np.percentile(vals, 100.0*cdf[0], axis=axis),
             np.percentile(vals, 100.0*cdf[1], axis=axis)]
            for cdf in cdf_vals]
    conf = np.array(conf)
    med = np.percentile(vals, 50.0, axis=axis)

    if(len(conf) == 1): conf = conf[0]

    return med, conf


def confidenceBands(xx, yy, xbins=10, xscale='lin', confInt=[0.68, 0.95], filter=None):
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
    if(not np.iterable(confInt)):
        squeeze = True
        confInt = [confInt]
    xx = np.asarray(xx).flatten()
    yy = np.asarray(yy).flatten()
    if(xx.shape != yy.shape):
        errStr = "Shapes of `xx` and `yy` must match ('{}' vs. '{}'."
        errStr = errStr.format(str(xx.shape), str(yy.shape))
        raise ValueError(errStr)

    # Filter based on whether `yy` values match `filter` comparison to 0.0
    if filter is not None:
        compFunc = _comparisonFunction(filter)
        inds = np.where(compFunc(yy, 0.0))[0]
        xx = xx[inds]
        yy = yy[inds]

    # Create bins
    xbins = asBinEdges(xbins, xx)
    nbins = xbins.size - 1
    # Find the entries corresponding to each bin
    groups = groupDigitized(xx, xbins[1:], edges='right')
    # Allocate storage for results
    med = np.zeros(nbins)
    conf = np.zeros((nbins, np.size(confInt), 2))
    count = np.zeros(nbins, dtype=int)

    # Calculate medians and confidence intervals
    for ii, gg in enumerate(groups):
        count[ii] = np.size(gg)
        if(count[ii] == 0): continue
        mm, cc = confidenceIntervals(yy[gg], ci=confInt)
        med[ii] = mm
        conf[ii, ...] = cc[...]

    if squeeze:
        conf = conf.squeeze()

    return count, med, conf, xbins


def frexp10(vals):
    """Return the mantissa and exponent in base 10

    Arguments
    ---------
        vals <flt>(N) : values to be converted

    Returns
    -------
        man <flt>(N) : mantissa
        exp <flt>(N) : exponent

    """

    exp = np.int(np.floor(np.log10(vals)))
    man = vals / np.power(10.0, exp)
    return man, exp


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


def groupDigitized(arr, bins, edges='right'):
    """Get a list of array indices corresponding to each bin.

    Uses ``numpy.digitize`` to find which bin each element of ``arr`` belongs in.  Then, for each
    bin, finds the list of array indices which belong in that bin.

    Arguments
    ---------
        arr : array_like of scalars,
            Values to digitize and group.
        bins : array_like or scalars, shape (N,)
            Bin edges to digitize and group by.
        edges : str, {'right', 'left'}
            Whether bin edges correspond to 'right' or 'left' side of the bins.

    Returns
    -------
        groups : list of int arrays, shape (N,)
            Each list contains the ``arr`` indices belonging to each corresponding bin.


    Examples
    --------
        >>> arr = [ 0.0, 1.3, 1.8, 2.1 ]
        >>> bins = [ 1.0, 2.0, 3.0 ]
        >>> zcode.Math.groupDigitized(arr, bins, right=True)
        [array([0]), array([1, 2]), array([3])]
        >>> zcode.Math.groupDigitized(arr, bins, right=False)
        [array([1, 2]), array([3]), array([])]

    See Also
    --------
    -   ``scipy.stats.binned_statistic``
    -   ``numpy.digitize``

    """

    if(edges.startswith('r')): right = True
    elif(edges.startswith('l')): right = False
    else: RuntimeError("``edges`` must be 'right' or 'left'!")

    # `numpy.digitize` always assumes `bins` are right-edges (in effect)
    shift = 0
    # If we want 'left' bin edges, such shift each bin leftwards
    if(not right): shift = -1

    # Find in which bin each element of arr belongs
    pos = np.digitize(arr, bins, right=right) + shift

    groups = []
    # Group indices by bin number
    for ii in xrange(len(bins)):
        groups.append(np.where(pos == ii)[0])

    return groups


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
    if(log):
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
    levels = spacing(yp, scale='lin', num=num)
    samples = interpBack(levels)

    # Convert back to normal space, as needed
    if(log): samples = np.power(10.0, samples)

    if(sort): samples = samples[np.argsort(samples)]

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
        width <obj>    : scalar specifying the region to be smoothed, of twp values are given
                         they are taken as left and right bounds
        loc   <flt>    : int or float specifying to center position of smoothing,
                         ``width`` is used relative to this position, if provided.
        mode  <str>    : type of convolution, passed to ``numpy.convolve``

    Returns
    -------
        smArr <flt>[N] : smoothed array

    """

    length = np.size(arr)
    size = _fracToInt(size, length, within=1.0, round='floor')

    assert size <= length, "``size`` must be less than length of input array!"

    window = np.ones(int(size))/float(size)

    # Smooth entire array
    smArr = np.convolve(arr, window, mode=mode)

    # Return full smoothed array if no bounds given
    if(width is None):
        return smArr

    # Other convolution modes require dealing with differing lengths
    #    If smoothing only a portion of the array,
    assert mode == 'same', "Other convolution modes not supported for portions of array!"

    # Smooth portion of array
    # -----------------------

    if(np.size(width) == 2):
        lef = width[0]
        rit = width[1]
    elif(np.size(width) == 1):
        if(loc is None): raise ValueError("For a singular ``width``, ``pos`` must be provided!")
        lef = width
        rit = width
    else:
        raise ValueError("``width`` must be one or two scalars!")

    # Convert fractions to positions, if needed
    lef = _fracToInt(lef, length-1, within=1.0, round='floor')
    rit = _fracToInt(rit, length-1, within=1.0, round='floor')

    # If ``loc`` is provided, use ``width`` relative to that
    if(loc is not None):
        loc = _fracToInt(loc, length-1, within=1.0, round='floor')
        lef = loc - lef
        rit = loc + rit

    mask = np.ones(length, dtype=bool)
    mask[lef:rit] = False
    smArr[mask] = arr[mask]

    return smArr


def mono(arr, type='g', axis=-1):
    """Check for monotonicity in the given array.

    Arguments
    ---------
    arr : array_like scalars
        Input array to check.
    type : str
        Type of monotonicity to look for:
        * 'g' :

    Returns
    -------
    retval : bool
        Whether the input array is monotonic in the desired sense.

    """
    good_type = ['g', 'ge', 'l', 'le', 'e']
    assert type in good_type, "Type '%s' Unrecognized." % (type)
    # Retrieve the numpy comparison function (e.g. np.greater) for the given `type` (e.g. 'g')
    func = _comparisonFunction(type)
    delta = np.diff(arr, axis=axis)
    retval = np.all(func(delta, 0.0))
    return retval


def _comparisonFunction(comp):
    """Retrieve the comparison function matching the input expression.
    """
    if(comp == 'g'):
        func = np.greater
    elif(comp == 'ge'):
        func = np.greater_equal
    elif(comp == 'l'):
        func = np.less
    elif(comp == 'le'):
        func = np.less_equal
    elif(comp == 'e'):
        func = np.equal
    else:
        raise ValueError("Unrecognized comparison '%s'." % (comp))

    return func


def _fracToInt(frac, size, within=None, round='floor'):
    """Convert from a float ``frac`` to that fraction of ``size``.

    If ``frac`` is already an integer, do nothing.

    Arguments
    ---------
        frac   <flt>  : fraction to convert
        size   <int>  : find the fraction of this size
        within <obj>  : assert that ``frac`` is within [0.0,``within``], `None` for no assertion
        round  <str>  : which direction to round {'floor','ceil'}

    Returns
    -------
        loc    <int>  : ``frac`` of ``size`` as rounded integer

    """
    # If ``frac`` is already an integer, do nothing, return it
    if(isinstance(frac, numbers.Integral)): return frac

    if(round == 'floor'):
        roundFunc = np.floor
    elif(round == 'ceil'):
        roundFunc = np.ceil
    else:
        raise ValueError("Unrecognized ``round``!")

    # Convert fractional input into an integer
    if(within is not None):
        assert frac >= 0.0 and frac <= within, "``frac`` must be between [0.0,%s]!" % (str(within))

    loc = np.int(roundFunc(frac*size))

    return loc


'''
def createSlice(index, max):
    """
    Create an array slicing object.

    Arguments
    ---------
        index <obj> : int, list of int, or 'None'
        max   <int> : length of array to be sliced.

    Returns
    -------
        ids <int>([N]) : indices included in slice.
        cut <obj>      : Slicing object, either `slice` or `np.array`.

    """

    import numbers
    # Single Value
    if(isinstance(index, numbers.Integral)):
        cut = slice(index, index+1)
        ids = np.arange(index, index+1)
    # Range of values
    elif(np.iterable(index)):
        cut = index
        ids = index
    else:
        if(index is not None):
            self.log.error("Unrecognized `index` = '%s'!" % (str(index)))
            self.log.warning("Returning all entries")

        cut = slice(None)
        ids = np.arange(max)


    return ids, cut

# } createSlice()
'''
