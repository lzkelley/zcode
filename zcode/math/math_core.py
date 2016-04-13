"""General functions for mathematical and numerical operations.

Functions
---------
-   contiguousInds           - Find the largest segment of contiguous array values
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
-   renumerate               - construct a reverse enumeration iterator.
-   frexp10                  - Decompose a float into mantissa and exponent (base 10).
-   groupDigitized           - Get a list of array indices corresponding to each bin.
-   mono                     - Check for monotonicity in the given array.
-   round                    -

-   _comparisonFunction      -
-   _comparisonFilter        -
-   _fracToInt               -
-   _infer_scale             -

"""
from __future__ import absolute_import, division, print_function, unicode_literals
from six.moves import xrange

import numpy as np
import warnings
import numbers

__all__ = ['contiguousInds', 'within', 'indsWithin', 'minmax', 'really1d',
           'argextrema', 'spacing', 'asBinEdges', 'strArray', 'sliceForAxis', 'midpoints',
           'vecmag', 'renumerate', 'frexp10', 'groupDigitized',
           'mono', 'ordered_groups', 'around', 'comparison_filter',
           '_comparisonFunction', '_comparison_function', '_infer_scale']


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


def minmax(data, prev=None, stretch=0.0, filter=None, limit=None, round=None):
    """Find minimum and maximum of given data, return as numpy array.

    If ``prev`` is provided, the returned minmax values will also be compared to it.
    To compare with only a previous minimum or maximum, pass the other as `None`, e.g.
    ``prev = [None, 2.5]`` to compare only with a maximum of `2.5`.

    Arguments
    ---------
    data : [...] ndarray of scalar of any shape
        Arbitrarily shaped data to find minumum and maximum of.
    prev : `None` or (2,) array_like of scalar and/or `None`
        Also find min/max against prev[0] and prev[1] respectively.  If `prev` is `None`,
        or if either of the elements of `prev` are `None`, then they are not compared.
    filter : str or `None`,
        Key describing how to filter the input data, or `None` for no filter.
        See, ``comparison_filter``.
    stretch : flt
        Factor by which to stretch min and max by (``1.0 +- stretch``).
    limit :
    round : int or 'None'
        The number of significant figures to which to round the min and max values.

    Returns
    -------
    minmax : (2,) array of scalar, or `None`
        Minimum and maximum of given data (and ``prev`` if provided).  If the input data is empty,
        then `prev` is returned --- even if that is `None`.

    To-Do
    -----
    -   Add an 'axis' argument.

    """
    if prev is not None:
        assert len(prev) == 2, "`prev` must have length 2."
    if limit is not None:
        assert len(limit) == 2, "`limit` must have length 2."

    useData = np.array(data)
    if filter:
        useData = comparison_filter(useData, filter)

    # If there are no elements (left), return `prev` (`None` if not provided)
    if np.size(useData) == 0:
        return prev

    # Determine stretch factor
    lef = (1.0-stretch)
    rit = (1.0+stretch)

    # Find extrema
    minmax = np.array([lef*np.min(useData), rit*np.max(useData)])

    # Compare to previous extrema, if given
    if prev is not None:
        if prev[0] is not None: minmax[0] = np.min([minmax[0], prev[0]])
        if prev[1] is not None: minmax[1] = np.max([minmax[1], prev[1]])

    # Compare to limits, if given
    if limit is not None:
        if limit[0] is not None: minmax[0] = np.max([minmax[0], limit[0]])
        if limit[1] is not None: minmax[1] = np.min([minmax[1], limit[1]])

    # Round the min/max results to given number of sig-figs
    if round is not None:
        minmax[0] = around(minmax[0], round, 'log', 'floor')
        minmax[1] = around(minmax[1], round, 'log', 'ceil')

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


def spacing(data, scale='log', num=100, filter=None, nonzero=None, positive=None):
    """Create an evenly spaced array between extrema from the given data.

    If ``nonzero`` and ``positive`` are not given, educated guesses are made based on ``scale``.

    Arguments
    ---------
    data : array_like of scalar
        Data from which to extract the extrema for bounds.
    scale : str
        Scaling for spacing, {'lin', 'log'}.
    num : int
        Number of points to produce, `N`.
    filter : str or `None`
        String specifying how to filter the input `data` relative to zero.
    [Deprecated]
        nonzero : bool
            Only use ``!= 0.0`` elements of `data`.
        positive : bool
            Only use ``>= 0.0`` elements of `data`.

    Returns
    -------
       spacing <scalar>[N] : array of evenly spaced points, with number of elements ``N = num``

    """
    if scale.startswith('log'):
        log_flag = True
    elif scale.startswith('lin'):
        log_flag = False
    else:
        raise RuntimeError("``scale`` '%s' unrecognized!" % (scale))

    # ---- DEPRECATION SECTION -------
    filter = _flagsToFilter(positive, nonzero, filter=filter, source='spacing')
    # --------------------------------

    # If no `filter` is given, and we are log-scaling, use a ``> 0.0`` filter
    if filter is None and log_flag:
        filter = '>'

    span = minmax(data, filter=filter)
    if log_flag:
        spacing = np.logspace(*np.log10(span), num=num)
    else:
        spacing = np.linspace(*span, num=num)

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

    if first is None or last is None:
        first = None
        last = 0

    # Create the style specification
    form = "{:%s}" % (format)

    arrStr = "["
    # Add the first `first` elements
    if first or first is None:
        arrStr += delim.join([form.format(vv) for vv in arr[:first]])

    # Include separator unless full array is being printed
    if first is not None:
        arrStr += "... "

    # Add the last `last` elements
    if last:
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


def renumerate(arr):
    """
    Same as ``enumerate`` but in reverse order.  Uses iterators, no copies made.
    """
    return zip(reversed(range(len(arr))), reversed(arr))


def frexp10(vals):
    """Return the mantissa and exponent in base 10.

    Arguments
    ---------
    vals : (N,) array_like of float
        Values to be converted.

    Returns
    -------
    man : (N,) array_like of float
        Mantissa.
    exp : (N,) array_like of float
        Exponent
    """
    # Find exponent of absolute value
    exp = np.floor(np.log10(np.fabs(vals)))
    # Positive/negative is still included here
    man = vals / np.power(10.0, exp)
    return man, exp


def groupDigitized(arr, bins, edges='right'):
    """Get a list of array indices corresponding to each bin.

    Uses ``numpy.digitize`` to find which bin each element of ``arr`` belongs in.  Then, for each
    bin, finds the list of array indices which belong in that bin.

    The specification for `bins` can be either monotonically increasing or decreasing, following
    the rules of ``numpy.digitize`` (which is used internally).

    Arguments
    ---------
    arr : array_like of scalars
        Values to digitize and group.
    bins : (N,) array_like or scalars
        Bin edges to digitize and group by.
    edges : str, {'[r]ight', '[l]eft'}
        Whether bin edges correspond to 'right' or 'left' side of the bins.
        Only the first letter is tested, and either case (lower/upper) is allowed.

    Returns
    -------
    groups : (N,) list of int arrays
        Each list contains the ``arr`` indices belonging to each corresponding bin.


    Examples
    --------
        >>> arr = [0.0, 1.3, 1.8, 2.1]
        >>> bins = [1.0, 2.0, 3.0]
        >>> zcode.Math.groupDigitized(arr, bins, right=True)
        [array([0]), array([1, 2]), array([3])]
        >>> zcode.Math.groupDigitized(arr, bins, right=False)
        [array([1, 2]), array([3]), array([])]

    See Also
    --------
    -   ``scipy.stats.binned_statistic``
    -   ``numpy.digitize``

    """
    edges = edges.lower()
    if edges.startswith('r'): right = True
    elif edges.startswith('l'): right = False
    else: RuntimeError("``edges`` must be 'right' or 'left'!")

    # `numpy.digitize` always assumes `bins` are right-edges (in effect)
    shift = 0
    # If we want 'left' bin edges, such shift each bin leftwards
    if not right: shift = -1

    # Find in which bin each element of arr belongs
    pos = np.digitize(arr, bins, right=right) + shift

    groups = []
    # Group indices by bin number
    for ii in xrange(len(bins)):
        groups.append(np.where(pos == ii)[0])

    return groups


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
    arr = np.atleast_1d(arr)
    if arr.size == 1: return True
    good_type = ['g', 'ge', 'l', 'le', 'e']
    assert type in good_type, "Type '%s' Unrecognized." % (type)
    # Retrieve the numpy comparison function (e.g. np.greater) for the given `type` (e.g. 'g')
    func = _comparisonFunction(type)
    delta = np.diff(arr, axis=axis)
    retval = np.all(func(delta, 0.0))
    return retval


def ordered_groups(values, targets, inds=None, dir='above', include=False):
    """Find the locations in ordering indices to break the given values into target groups.

    Arguments
    ---------
    values : (N,) array_like of scalar
        Values to order by.
    targets : scalar or (M,) array_like of scalar
        Target locations at which to divide `values` into groups.
    inds : (L,) array_like of int or `None`
        Subset of elements in the input `values` to consider.
    dir : str {'a', 'b'}
        Consider elements 'a'bove or 'b'elow each `targets` value.
    include : bool
        Include the `targets` values themselves in the groups.
        See: Example[1]

    Returns
    -------
    locs : int or (M,) array of int
        Locations in the array of sorting indices `sorter` at which to divide groups.
        If the given `targets` is a scalar, then the returned `locs` is also.
    sorter : (N,) [or (L,) if `inds` is provided] of int
        Indices to sort the input array `values`, [or `values[inds]` if `inds` is provided].

    Examples
    --------
    # [1] Inclusive vs. Exclusive  and  Scalar vs. array_like input
    >>> ordered_groups([5, 4, 3, 2], 3, dir='b', include=False)
    1
    >>> ordered_groups([5, 4, 3, 2], [3], dir='b', include=True)
    [2]

    """
    values = np.asarray(values)
    nval = values.size
    scalar = False
    if np.ndim(targets) == 0:
        scalar = True
    targets = np.atleast_1d(targets)
    if dir.startswith('a'):
        above = True
        if include: side = 'left'
        else: side = 'right'
    elif dir.startswith('b'):
        above = False
        if include: side = 'right'
        else: side = 'left'
    else:
        raise ValueError("`dir` = '{}' must be either 'a'bove or 'b'elow.".format(dir))

    if not mono(targets):
        raise ValueError("`targets` must be in increasing order.")

    if inds is None:
        inds = np.arange(nval)

    # Find indices to sort by `mass_ratio`
    sorter = np.argsort(values[inds])
    # Find the locations in the sorted array at which the desired `mrats` occur.
    locs = np.searchsorted(values[inds], targets, sorter=sorter, side=side)
    # Reverse sorter so to get elements ABOVE target mass-ratios
    if above:
        sorter = sorter[::-1]
        locs = inds.size - locs

    if scalar:
        locs = np.asscalar(locs)

    return locs, sorter


def around(val, decimals=0, scale='log', dir='near'):
    """Round the given value to arbitrary decimal points, in any direction.

    Perhaps rename `scale` to `sigfigs` or something?  Not really in 'log' scaling...

    Arguments
    ---------
    val : scalar
        Value to be rounded.
    decimals : int
        Number of decimal places at which to round.
    scale : str, {'log', 'lin'}
        How to interpret the number of decimals/precision at which to round.
        +   'log': round to `decimals` number of significant figures.
        +   'lin': round to `decimals` number of decimal points.
    dir : str, {'near', 'ceil', 'floor'}
        Direction in which to round.
        +   'nearest': use `np.around` to round the nearest 'even' value.
        +   'ceil': use `np.ceil` to round to higher (more positive) values.
        +   'floor': use `np.floor` to round to lower (more negative) values.

    Returns
    -------
    rnd : scalar
        Rounded version of the input `val`.

    """
    from zcode.plot import plot_core
    islog = plot_core._scale_to_log_flag(scale)
    if islog and decimals < 0:
        err_str = "With 'log' scaling and `decimals` = '{}' < 0, all results zero.".format(decimals)
        raise ValueError(err_str)
    if np.size(val) > 1:
        raise ValueError("Arrays are not yet supported.")

    if islog:
        useval, exp = frexp10(val)
    else:
        useval = np.array(val)
        exp = 0.0

    dpow = np.power(10.0, decimals)

    # Round to nearest ('n'earest)
    if dir.startswith('n'):
        useval = np.around(useval, decimals)
    # Round up ('c'eiling)
    elif dir.startswith('c') or dir.startswith('u'):
        useval *= dpow
        useval = np.ceil(useval)
        useval /= dpow
    # Round down ('f'loor)
    elif dir.startswith('f') or dir.startswith('d'):
        useval *= dpow
        useval = np.floor(useval)
        useval /= dpow
    else:
        raise ValueError("Given `dir` = '{}' not supported.".format(dir))

    rnd = useval * np.power(10.0, exp)
    return rnd


def _comparisonFunction(comp):
    """[DEPRECATED]Retrieve the comparison function matching the input expression.
    """
    # ---- DECPRECATION SECTION ----
    warnStr = ("Using deprecated function '_comparisonFunction'.  "
               "Use '_comparison_function' instead.")
    warnings.warn(warnStr, DeprecationWarning, stacklevel=3)
    # ------------------------------

    if comp == 'g' or comp == '>':
        func = np.greater
    elif comp == 'ge' or comp == '>=':
        func = np.greater_equal
    elif comp == 'l' or comp == '<':
        func = np.less
    elif comp == 'le' or comp == '<=':
        func = np.less_equal
    elif comp == 'e' or comp == '=' or comp == '==':
        func = np.equal
    elif comp == 'ne' or comp == '!=':
        func = np.not_equal
    else:
        raise ValueError("Unrecognized comparison '%s'." % (comp))

    return func


def _comparison_function(comp, value=0.0):
    """Retrieve the comparison function matching the input expression.

    Arguments
    ---------
    comp : str
        Key describing the type of comparison.
    value : scalar
        Value with which to compare.

    Returns
    -------
    comp_func : callable
        Comparison function which returns a or an-array-or bool matching the input
        shape, describing how the input values compare  to `value`.

    """
    if comp == 'g' or comp == '>':
        func = np.greater
    elif comp == 'ge' or comp == '>=':
        func = np.greater_equal
    elif comp == 'l' or comp == '<':
        func = np.less
    elif comp == 'le' or comp == '<=':
        func = np.less_equal
    elif comp == 'e' or comp == '=' or comp == '==':
        func = np.equal
    elif comp == 'ne' or comp == '!=':
        func = np.not_equal
    else:
        raise ValueError("Unrecognized comparison '{}'.".format(comp))

    def comp_func(xx):
        return func(xx, value)

    return comp_func


def _comparisonFilter(data, filter):
    """
    """
    # ---- DECPRECATION SECTION ----
    warnStr = ("Using deprecated function '_comparisonFilter'.  "
               "Use '_comparison_filter' instead.")
    warnings.warn(warnStr, DeprecationWarning, stacklevel=3)
    # ------------------------------
    if filter is None:
        return data
    if not callable(filter):
        filter = _comparisonFunction(filter)
    sel = np.where(filter(data, 0.0) & np.isfinite(data))
    return data[sel]


def comparison_filter(data, filter, inds=False, value=0.0, finite=True):
    """
    """
    if filter is None:
        return data
    if not callable(filter):
        filter = _comparison_function(filter, value=value)

    # Include is-finite check
    if finite:
        sel = np.where(filter(data) & np.isfinite(data))
    else:
        sel = np.where(filter(data))

    if inds:
        return sel
    else:
        return np.asarray(data)[sel]


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


def _flagsToFilter(positive, nonzero, filter=None, source=None):
    """Function to convert from (deprecated) flags to a (new-style) `filter` string.

    Example:
        # ---- DEPRECATION SECTION -------
        filter = _flagsToFilter(positive, nonzero, filter=filter, source='minmax')
        # --------------------------------

    """

    # Warn if using deprecated arguments
    warnStr = ''
    if source:
        warnStr += '`{:s}`: '.format(str(source))
    warnStr += "Using deprecated parameter `{:s}`; use `filter` instead."
    if positive is not None:
        warnings.warn(warnStr.format('positive'), DeprecationWarning, stacklevel=3)
    if nonzero is not None:
        warnings.warn(warnStr.format('nonzero'), DeprecationWarning, stacklevel=3)

    # Convert arguments into a `filter` str
    if positive or nonzero:
        if filter is not None:
            raise ValueError("Cannot use `positive`/`nonzero` with `filter`.")
        filter = ''
        if positive:
            filter += '>'
            if not nonzero:
                filter += '='
        elif nonzero:
            filter = '!='

    return filter


def _infer_scale(args):
    if np.all(args > 0.0): return 'log'
    return 'lin'


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
