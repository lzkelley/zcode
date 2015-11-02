"""General functions for mathematical and numerical operations.

Functions
---------
-   spline                               - Create a general spline interpolation function.
-   contiguousInds                       - Find the largest segment of contiguous array values
-   cumtrapz_loglog                      - Perform a cumulative integral in log-log space.
-   within                               - Test whether a value is within the bounds of another.
-   minmax                               - Find the min and max of given values.
-   spacing                              - Create an even spacing between extrema from given data.
-   strArray                             - Create a string representation of a numerical array.
-   sliceForAxis                         - Array slicing object which slices only the target axis.
-   midpoints                            - Return the midpoints between values in the given array.
-   vecmag                               - find the magnitude/distance of/between vectors.
-   extend                               - Extend the given array by extraplation.
-   renumerate                           - construct a reverse enumeration iterator.
-   cumstats                             - Calculate a cumulative average and standard deviation.
-   confidenceIntervals                  - Compute the values bounding desired confidence intervals.
-   frexp10                              - Decompose a float into mantissa and exponent (base 10).
-   stats                                - Get basic statistics for the given array.
-   groupDigitized                       - Find groups of array indices corresponding given bin.
-   sampleInverse                        - Find x-sampling to evenly divide a function in y-space.
-   smooth                               - Use convolution to smooth the given array.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import itertools
import numpy as np
import scipy as sp
import scipy.interpolate
import warnings
import numbers

__all__ = ['spline', 'contiguousInds', 'cumtrapz_loglog',
           'within', 'minmax', 'spacing', 'strArray', 'sliceForAxis', 'midpoints', 
           'vecmag', 'extend',
           'renumerate', 'cumstats', 'confidenceIntervals', 'frexp10', 'stats', 'groupDigitized',
           'sampleInverse', 'smooth']


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


def midpoints(arr, log=False, frac=0.5):
    """
    Return the midpoints between values in the given array.

    If the given array is N-dimensional, midpoints are calculated from the last dimension.

    Arguments
    ---------
        arr <flt>[...,N] : input array of length `N`
        log <bool>       : find midpoints in log of ``arr``
        frac <flt>       : fraction of the way between intervals

    Returns
    -------
        mids <flt>[...,N-1]: midpoints of length `N-1`

    """

    if(np.shape(arr)[-1] < 2):
        raise RuntimeError("Input ``arr`` does not have a valid shape!")

    if(log): user = np.log10(arr)
    else:      user = np.array(arr)

    diff = np.diff(user)

    # skip the last element, or the last axis
    cut = sliceForAxis(user, axis=-1, stop=-1)
    start = user[cut]
    mids = start + frac*diff

    if(log): mids = np.power(10.0, mids)

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


def confidenceIntervals(vals, ci=[0.68, 0.95, 0.997]):
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
    conf = [[np.percentile(vals, 100.0*cdf[0]), np.percentile(vals, 100.0*cdf[1])]
            for cdf in cdf_vals]
    conf = np.array(conf)
    med = np.percentile(vals, 50.0)

    if(len(conf) == 1): conf = conf[0]

    return med, conf


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
    """Find groups of array indices corresponding given bin.

    Uses ``numpy.digitize`` to find which bin each element of ``arr`` belongs in.  Then, for each
    bin, finds the list of array indices which belong in that bin.

    Example:
        >>> arr = [ 0.0, 1.3, 1.8, 2.1 ]
        >>> bins = [ 1.0, 2.0, 3.0 ]
        >>> zcode.Math.groupDigitized(arr, bins, right=True)
        [array([0]), array([1, 2]), array([3])]
        >>> zcode.Math.groupDigitized(arr, bins, right=False)
        [array([1, 2]), array([3]), array([])]

    Arguments
    ---------
        arr   <flt>[N] : array of values to digitize and group
        bins  <flt>[M] : array of bin edges to digitize and group by
        edges <str>    : whether bin edges are 'right' or 'left' {'right', 'left'}

    Returns
    -------
        groups <int>[M][...] : Each list contains the ``arr`` indices belonging in each bin

    """

    if(edges.startswith('r')): right = True
    elif(edges.startswith('l')): right = False
    else: RuntimeError("``edges`` must be 'right' or 'left'!")

    # ``numpy.digitize`` always assumes ``bins`` are right-edges (in effect)
    shift = 0
    # If we want 'left' bin edges, such shift each bin leftwards
    if(not right): shift = -1

    # Find in which bin each element of arr belongs
    pos = np.digitize(arr, bins, right=right) + shift

    groups = []
    # Group indices by bin number
    for ii in range(len(bins)):
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


def _fracToInt(frac, size, within=None, round='floor'):
    """
    Convert from a float ``frac`` to that fraction of ``size``.

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
