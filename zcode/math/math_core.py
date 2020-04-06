"""General functions for mathematical and numerical operations.

Functions
---------
-   argextrema               - Find the index of the extrema in the input array.
-   argnearest               - Find the indices in one array closest to those in a second array.
-   around                   - Round the given value to arbitrary decimal points, in any direction.
-   contiguousInds           - Find the largest segment of contiguous array values
-   within                   - Test whether a value is within the bounds of another.
-   indsWithin               - Find the indices within the given extrema.
-   minmax                   - Find the min and max of given values.
-   really1d                 - Test whether an array_like is really 1D (e.g. not jagged array).
-   spacing                  - Create an even spacing between extrema from given data.
-   asBinEdges               - Create bin-edges if the given `bins` do not already give them.
-   str_array                 - Create a string representation of a numerical array.
-   sliceForAxis             - Array slicing object which slices only the target axis.
-   midpoints                - Return the midpoints between values in the given array.
-   vecmag                   - find the magnitude/distance of/between vectors.
-   renumerate               - construct a reverse enumeration iterator.
-   frexp10                  - Decompose a float into mantissa and exponent (base 10).
-   groupDigitized           - Get a list of array indices corresponding to each bin.
-   mono                     - Check for monotonicity in the given array.
-   limit                    -
-   interp_func

-   _comparisonFunction      -
-   _comparisonFilter        -
-   _fracToInt               -
-   _infer_scale             -

"""
from __future__ import absolute_import, division, print_function, unicode_literals
import warnings
import numbers
import logging
# import six

import numpy as np
import scipy as sp
import scipy.interpolate  # noqa


__all__ = [
    'argextrema', 'argfirst', 'argfirstlast', 'arglast', 'argnearest',
    'around', 'array_str', 'asBinEdges',
    'broadcast', 'broadcastable', 'contiguousInds', 'edges_from_cents',
    'frexp10', 'groupDigitized', 'slice_with_inds_for_axis',
    'indsWithin', 'midpoints', 'minmax',  'mono', 'limit',
    'ordered_groups', 'really1d', 'renumerate', 'roll',

    'rotation_matrix_between_vectors', 'rotation_matrix_about',
    'xyz_to_rpt', 'rpt_to_xyz', 'xyz_to_rtp', 'rtp_to_xyz',

    'sliceForAxis', 'spacing', 'spacing_composite',
    'str_array', 'str_array_neighbors', 'str_array_2d', 'vecmag', 'within', 'zenumerate',
    'comparison_filter', '_comparison_function',
    '_infer_scale', '_fracToInt',
    # DEPRECATED
    'zenum'
]

from zcode import utils


def argextrema(arr, type, filter=None):
    """Find the index of the desired type of extrema in the input array.
    """
    warnings.warn("This function sucks!  Dont use it!")

    # Valid filters, NOTE: do *not* include 'e' (equals), doesn't make sense here.
    good_filter = [None, 'g', 'ge', 'l', 'le']
    if filter not in good_filter:
        raise ValueError("Filter '%s' Unrecognized." % (type))
    # Make sure `type` is valid
    good_type = ['min', 'max']
    if not np.any([type.startswith(gt) for gt in good_type]):
        raise ValueError("Type '%s' Unrecognized." % (type))
    # Make sure input array `arr` is valid (1D)
    arr = np.asarray(arr)
    if (arr.ndim != 1) or (arr[0].ndim != 0):
        raise ValueError("Only 1D arrays currently supported.")

    if type.startswith('min'):
        func = np.argmin
    elif type.startswith('max'):
        func = np.argmax

    # Find whether the `filter` criteria is True
    if filter:
        filter_func = _comparison_function(filter, 0.0)
        sel = filter_func(arr)
    # If no filter (`None`), all values are valid
    else:
        sel = np.ones_like(arr, dtype=bool)

    # Find the extrema within the valid (`sel`) subset
    ind = func(arr[sel])
    # Convert to index wrt the full input array
    ind = np.where(sel)[0][ind]
    return ind


def argfirst(arr, check=True, **kwargs):
    """Return the index of the first true element of the given array.
    """
    if check and not np.any(arr):
        raise ValueError("No elements of given array are true!")
    return np.argmax(arr, **kwargs)


def arglast(arr, check=True):
    """Return the index of the last true element of the given array.
    """
    if np.ndim(arr) != 1:
        raise ValueError("`arglast` not yet supported for ND != 1 arrays!")
    if check and not np.any(arr):
        raise ValueError("No elements of given array are true!")
    size = arr.size - 1
    return size - np.argmax(arr[::-1])


def argfirstlast(arr, **kwargs):
    """Return the indices of the first and last true elements of the given array.
    """
    return argfirst(arr), arglast(arr)


def argnearest(edges, vals, assume_sorted=False, side=None):
    """Find the indices of elements in the `edges` array closest to those in the `vals` array.

    Arguments
    ---------
    edges : (N,) array of scalar
        Find indices for elements in this array.
    vals : (M,) array of scalar
        Look for elements in the `edges` array closest to these `vals` values.
    assume_sorted : bool,
        Assume the input array of `edges` is sorted.
        (Note: `vals` can be unsorted regardless)
    side : str or `None`

    Returns
    -------
    idx : (M,) array of int
        Indices of `edges` nearest `vals`.  May include duplicates.

    """
    edges = np.atleast_1d(edges)
    scalar = np.isscalar(vals)
    vals = np.atleast_1d(vals)

    if edges.ndim != 1:
        raise ValueError("`edges` must be 1D!")

    shape = None
    # Flatten input array if it is multidimensional
    if vals.ndim > 1:
        shape = vals.shape
        size = np.product(shape)
        vals = vals.reshape(size)

    # Sort the input array if needed
    if not assume_sorted:
        srt = np.argsort(edges)
        edges = edges[srt]

    # Find the nearest bin on either side
    if side is None:
        # This is the index of the edge to the *right* of (i.e. above) each value
        idx = np.searchsorted(edges, vals, side="left").clip(max=edges.size-1)
        # Find the distances to each nearest bin edge
        dist_lo = np.fabs(vals - edges[idx-1])
        dist_hi = np.fabs(vals - edges[idx])
        # If left ('lo') is nearer, mask=1, and we shift from the right edge to the left edge
        mask = (idx > 0) & ((idx == edges.size) | (dist_lo < dist_hi))
        idx = idx - mask
    # If we want the 'r'ight nearest edge
    elif side.startswith('r'):
        # This is the index of the edge to the *right* of (i.e. above) each value
        #    WARNING: if a value is exactly on an edge, the *next* edge will be selected
        idx = np.searchsorted(edges, vals, side="right")
        # mask = (vals == edges[-1])
        # idx = idx + mask
    elif side.startswith('l'):
        # This is the index of the edge to the *right* of (i.e. above) each value
        #    WARNING: if a value is exactly on an edge, the *prev* edge will be selected
        idx = np.searchsorted(edges, vals, side="left") - 1
        # If a value matches the right-most edge, it should also be shifted to the prev edge,
        #    which requires an extra shift
        # mask = (vals == edges[-1])
        # idx = idx - mask
    else:
        raise ValueError("Unrecognized `side` argument '{}'!".format(side))

    # Reorder the indices if the input was unsorted
    #    NOTE: this is not re/un-sorting, this is changing the resulting numbers to reflect the
    #          input order of `edges`
    if not assume_sorted:
        # Leave out-of-bounds results (for 'left' and 'right') as the same value
        idx = [srt[ii] if (ii >= 0) and (ii < edges.size) else ii
               for ii in idx]

    if scalar:
        idx = idx[0]
    else:
        idx = np.asarray(idx)

    # Reshape output array if input was multidimensional
    if shape is not None:
        idx = idx.reshape(shape)

    return idx


def around(val, decimals=0, scale='log', dir='near'):
    """Round the given value to arbitrary decimal points, in any direction.

    Perhaps rename `scale` to `sigfigs` or something?  Not really in 'log' scaling...

    Arguments
    ---------
    val : scalar
        Value to be rounded.
    decimals : int
        Number of decimal places at which to round.
        If `scale` is 'log' and `decimals` is negative, then the nearest order of magnitude
        is returned, in the direction of `dir`.  NOTE: this rounding is done in log-space.
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
    if np.size(val) > 1:
        raise ValueError("Arrays are not yet supported.")

    # Round to nearest ('n'earest)
    if dir.startswith('n'):
        dir_int = 0
    # Round up ('c'eiling)
    elif dir.startswith('c') or dir.startswith('u'):
        dir_int = 1
    # Round down ('f'loor)
    elif dir.startswith('f') or dir.startswith('d'):
        dir_int = -1
    else:
        raise ValueError("Given `dir` = '{}' not supported.".format(dir))

    if islog:
        useval, exp = frexp10(val)
        # If `decimals` is negative and ``scale == 'log'``, round to order of magnitude
        # NOTE: this is done in log-space, i.e. 4.0e-4 rounds to nearest as 1e-4 (not 1e-3)
        if decimals < 0:
            useval = np.log10(val)
            # Round base-ten power in target direction
            if dir_int == 0:
                useval = np.around(useval, 0)
            elif dir_int == +1:
                useval = np.ceil(useval)
            elif dir_int == -1:
                useval = np.floor(useval)
            # Return order of magnitude
            return np.power(10.0, useval)
    else:
        useval = np.array(val)
        exp = 0.0

    dpow = np.power(10.0, decimals)

    # Round to nearest
    if dir_int == 0:
        useval = np.around(useval, decimals)
    # Round up
    elif dir_int == +1:
        useval *= dpow
        useval = np.ceil(useval)
        useval /= dpow
    # Round down
    elif dir_int == -1:
        useval *= dpow
        useval = np.floor(useval)
        useval /= dpow

    rnd = useval * np.power(10.0, exp)
    return rnd


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
    for i in range(len(smin)):
        if smin[i] == smax[i]:
            smin[i] = smin[i] - 0.5
            smax[i] = smax[i] + 0.5

    # Create arrays describing edges of bins
    for ii in range(ndim):
        if np.isscalar(bins[ii]):
            edges[ii] = spacing([smin[ii], smax[ii]], scale[ii], num=bins[ii] + 1)
        else:
            edges[ii] = np.asarray(bins[ii], float)

    if(ndim == 1):
        edges = edges[0]

    return edges


def broadcastable(*args):
    """Expand N, 1D arrays be able to be broadcasted into N, ND arrays.

    e.g. from arrays of len `3`,`4`,`2`, returns arrays with shapes: `3,1,1`, `1,4,1` and `1,1,2`.
    """
    ndim = len(args)
    assert np.all([1 == np.ndim(aa) for aa in args]), "Each array in `args` must be 1D!"

    cut_ref = [slice(None)] + [np.newaxis for ii in range(ndim-1)]
    cuts = [np.roll(cut_ref, ii).tolist() for ii in range(ndim)]
    outs = [aa[tuple(cc)] for aa, cc in zip(args, cuts)]
    return outs


def broadcast(*args):
    """Broadcast N, 1D arrays into N, ND arrays.

    e.g. from arrays of len `3`,`4`,`2`, returns arrays with shapes: `3,4,2`, `3,4,2` and `3,4,2`.

    NOTE: scalar arrays will not add a dimension!
    e.g. from array len `4` and scalar (i.e. len `0`) returns shapes: `4`, `4`

    """
    # Separate arguments into scalars and arrays
    idx = [np.isscalar(aa) for aa in args]
    sca_args = [aa for aa in args if np.isscalar(aa)]
    arr_args = [aa for aa in args if not np.isscalar(aa)]

    # If everything is a scalar, return them
    if len(arr_args) == 0:
        return args

    outs = np.meshgrid(*arr_args, indexing='ij')
    # If everything is an array, already done, return
    if not np.any(idx):
        return outs

    # Broadcast scalars to shape of broadcasted arrays
    sca_args = [aa * np.ones_like(outs[0]) for aa in sca_args]
    # Insert broadcasted scalars appropriately
    outs = np.insert(outs, np.where(idx)[0], sca_args, axis=0)

    return outs


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


def edges_from_cents(cents, scale='lin', log=None):
    if not any([scale.lower().startswith(sc) for sc in ['lin', 'log']]):
        raise ValueError("`scale` must be 'lin' or 'log'!")

    if log is None:
        log = scale.lower().startswith('log')

    if log:
        cents = np.log10(cents)

    widths = np.diff(cents)
    NUM = 5

    lo = sp.interpolate.PchipInterpolator(np.arange(NUM), widths[:NUM], extrapolate=True)(-1)
    hi = sp.interpolate.PchipInterpolator(np.arange(NUM), widths[-NUM:], extrapolate=True)(NUM)
    edges = midpoints(cents, log=False)
    edges = np.concatenate(([edges[0] - lo], edges, [edges[-1] + hi]))

    if log:
        edges = np.power(10.0, edges)

    return edges


def slice_with_inds_for_axis(arr, inds, axis):
    """Use an N-1 dimensional array with indices along a particular axis of an N dimensional array

    See: https://stackoverflow.com/a/46103129/230468
    """
    if axis < 0:
        axis += np.ndim(arr)

    new_shape = list(arr.shape)
    del new_shape[axis]

    grid = np.ogrid[tuple(map(slice, new_shape))]
    grid.insert(axis, inds)

    return arr[tuple(grid)]


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
    for ii in range(len(bins)):
        groups.append(np.where(pos == ii)[0])

    return groups


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


def midpoints(arr, scale=None, log=None, frac=0.5, axis=-1, squeeze=True):
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

    if (np.shape(arr)[axis] < 2):
        raise RuntimeError("Input ``arr`` does not have a valid shape!")

    if log is not None:
        if scale is not None:
            raise ValueError("Provide either `scale` or `log`, not both!")
        log_flag = log
    elif scale is not None:
        log_flag = scale.lower().startswith('log')
    # Default to log scaling
    else:
        log_flag = True

    # Convert to log-space
    if log_flag:
        user = np.log10(arr)
    else:
        user = np.array(arr)

    diff = np.diff(user, axis=axis)

    #     skip the last element, or the last axis
    cut = sliceForAxis(user, axis=axis, stop=-1)
    start = user[cut]
    mids = start + frac*diff

    if log_flag:
        mids = np.power(10.0, mids)
    if squeeze:
        mids = mids.squeeze()

    return mids


def minmax(data, prev=None, stretch=None, log_stretch=None, filter=None, limit=None,
           round=None, round_scale='log', type=None, percs=None):
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
    stretch : float, `None`, or (2,) or float
        Factor by which to stretch min and max by (``1.0 +- stretch``) in linear space.
        If two values are given, they are applied to low (-) and high (+) side respectively.
    log_stretch : float, `None`, or (2,) or float
        Factor by which to stretch min and max by (``1.0 +- stretch``) in log space.
        If two values are given, they are applied to low (-) and high (+) side respectively.
    limit :
    round : int or 'None'
        The number of significant figures to which to round the min and max values.
    round_scale : str, {'lin', 'log'}
        In which scaling to round in.

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

    # Pre-flatten the input data (NOTE: `comparison_filter` fails for jagged arrays!)
    try:
        # Check if input array is jagged (will raise an error)
        # np.product(data)   # NOTE: this causes overflow errors.
        np.linalg.norm(data)
        data = np.array(data).flatten()
    except ValueError:
        try:
            data = [np.array(dd).flatten() for dd in data]
            data = np.concatenate(data)
        except:
            logging.error("Failed to flatten input data!")
            raise

    if filter is not None:
        data = comparison_filter(data, filter)

    # If there are no elements (left), return `prev` (`None` if not provided)
    if np.size(data) == 0:
        msg = "" if filter is None else " after filtering"
        logging.warning("zcode.math.math_core:minmax() :: Empty `data` array{}!".format(msg))
        return prev

    # Find extrema
    if percs is None:
        minmax = np.array([np.min(data), np.max(data)])
    else:
        from zcode.math.statistic import percentiles
        assert np.size(percs) == 2, "Provided `percs` must be length two!"
        minmax = percentiles(data, percs)

    if type is not None:
        minmax = minmax.astype(type)

    # Add stretch (relative to center point)
    if (stretch is not None) or (log_stretch is not None):
        if (stretch is not None) and (log_stretch is not None):
            raise ValueError("Only `stretch` OR `log_stretch` can be applied!")

        # Choose the given value
        fact = stretch if (stretch is not None) else log_stretch

        # If a single stretch value is given, duplicate it for both lo and hi sides
        if (fact is not None) and np.isscalar(fact):
            fact = [fact, fact]
        # Make sure size is right
        if (fact is not None) and np.size(fact) != 2:
            raise ValueError("`log_stretch` and `stretch` must be None, scalar or (2,)!")

        # Use log-values as needed (stretching in log-space)
        _minmax = np.log10(minmax) if (log_stretch is not None) else minmax
        # Find the center, and stretch relative to that
        cent = np.average(_minmax)
        _minmax[0] = cent - (1.0 + fact[0])*(cent - _minmax[0])
        _minmax[1] = cent + (1.0 + fact[1])*(_minmax[1] - cent)
        # Convert back to normal-space as needed
        minmax = np.power(10.0, _minmax) if (log_stretch is not None) else _minmax

    # Compare to previous extrema, if given
    if prev is not None:
        if prev[0] is not None:
            minmax[0] = np.min([minmax[0], prev[0]])
        if prev[1] is not None:
            minmax[1] = np.max([minmax[1], prev[1]])

    # Compare to limits, if given
    if limit is not None:
        if limit[0] is not None:
            minmax[0] = np.max([minmax[0], limit[0]]) if not np.isnan(minmax[0]) else limit[0]
        if limit[1] is not None:
            minmax[1] = np.min([minmax[1], limit[1]]) if not np.isnan(minmax[1]) else limit[1]

    # Round the min/max results to given number of sig-figs
    if round is not None:
        minmax[0] = around(minmax[0], round, round_scale, 'floor')
        minmax[1] = around(minmax[1], round, round_scale, 'ceil')

    return minmax


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
    if arr.size == 1:
        return True
    good_type = ['g', 'ge', 'l', 'le', 'e']
    assert type in good_type, "Type '%s' Unrecognized." % (type)
    # Retrieve the numpy comparison function (e.g. np.greater) for the given `type` (e.g. 'g')
    func = _comparison_function(type, 0.0)
    delta = np.diff(arr, axis=axis)
    retval = np.all(func(delta))
    return retval


def limit(val, arr):
    """Limit the given value(s) to given bounds.

    Arguments
    ---------
    val : (N,) scalar or array of scalar
        Value(s) to be limited.
    arr : (M,) array of scalar
        The extrema by which to bound.

    Returns
    -------
    new : (N,) scalar or array of scalar
        Limited values, same size as input `val`.

    """
    # Make copy
    new = np.array(val)
    extr = minmax(arr)
    # Enforce lower bound
    new = np.maximum(new, extr[0])
    # Enforce upper bound
    new = np.minimum(new, extr[1])
    return new


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


def renumerate(arr):
    """
    Same as ``enumerate`` but in reverse order.  Uses iterators, no copies made.
    """
    return zip(reversed(range(len(arr))), reversed(arr))


def roll(a, r, cat=None, axis=-1):
    """Roll an array along a target axis by a varying amount.

    Arguments
    ---------
    a : array_like
        Array to be rolled
    r : array_like of int
        Amount to roll `a`.  Must be the same shape as `a`, without the target axis.
        e.g. if `a` has shape (N,M,L,) and `axis=1` then `r` must have shape (N,L,).

    From Stackoverflow @Divakar: https://stackoverflow.com/a/58474469/230468

    """

    import skimage

    if np.isscalar(r):
        # If `cat` is not None, then `np.roll` does not achieve the same thing
        if cat is None:
            return np.roll(a, r, axis=axis)
        else:
            r = r * np.ones_like(a)

    r = np.asarray(r)
    a = np.asarray(a)
    # If no concatenation array is provided, repeat a itself
    if cat is None:
        cat = a   # [..., :-1]

    if axis != -1:
        a = np.moveaxis(a, axis, -1)
        cat = np.moveaxis(cat, axis, -1)

    if a.shape[:-1] != r.shape:
        err = "Shape of `a` ({}) must match `r` ({}) except for target axis (axis={})!".format(
            a.shape, r.shape, axis)
        raise ValueError(err)

    # Repeate along target axis to accomodate any roll
    # a_ext = np.concatenate((a, a[..., :-1]), axis=-1)
    a_ext = np.concatenate((a, cat), axis=-1)

    # Get sliding windows; use advanced-indexing to select appropriate ones
    n = a.shape[-1]
    w = skimage.util.shape.view_as_windows(a_ext, (1,)*r.ndim + (n,)).reshape(a.shape + (n,))
    idx = (n - r) % n
    idxr = idx.reshape(idx.shape + (1,)*(w.ndim - r.ndim))

    res = np.take_along_axis(w, idxr, axis=r.ndim)[..., 0, :]

    if axis != -1:
        res = np.moveaxis(res, -1, axis)

    return res


def rotation_matrix_between_vectors(aa, bb):
    """Construct a rotation matrix that rotates vector `aa` to vector `bb`.
    """
    ab = np.matrix(aa + bb)
    rot = 2*((ab*ab.T) / (ab.T*ab)) - np.eye(3)
    return rot


def rotation_matrix_about(axis, theta):
    """Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.

    Taken from: https://stackoverflow.com/a/6802723

    """
    if np.shape(axis) != (3,):
        raise ValueError("Shape of `axis` must be (3,)!")

    scalar = True
    if np.ndim(theta) > 1:
        raise ValueError("Only 0 or 1 dimensional values for `theta` are supported!")
    elif np.ndim(theta) == 1:
        theta = np.atleast_2d(theta).T
        scalar = False

    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0).squeeze()
    # b, c, d = - axis * np.sin(theta / 2.0)
    temp = - axis * np.sin(theta / 2.0)
    if not scalar:
        temp = temp.T

    b, c, d = temp
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d

    rot = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                    [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                    [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

    if not scalar:
        rot = rot.T

    return rot


def xyz_to_rpt(xyz):
    """Convert from cartesian to spherical coordinates.

    Spherical is defined as ``(rad, phi, theta)``, where `phi` is the azimuthal angle (xy),
    and `theta` is the polar angle, with the +z-axis corresponding to ``theta = 0``.

    NOTE: 3-coordinates must be the zeroth axis.

    Arguments
    ---------
    xyz : (3, ...) array of scalar
        Cartesian coordinate vector.

    Returns
    -------
    rpt : (3, ...) array of scalar
        Spherical coordinate vector, same shape as input.

    """
    xyz = np.asarray(xyz)
    if (np.ndim(xyz) < 1) or (np.shape(xyz)[0] != 3):
        raise ValueError("`xyz` ({}) must have shape (3, ...)!".format(xyz.shape))

    rpt = np.zeros_like(xyz)
    xy = xyz[0, ...]**2 + xyz[1, ...]**2
    rpt[0, ...] = np.sqrt(xy + xyz[2, ...]**2)
    rpt[1, ...] = np.arctan2(xyz[1, ...], xyz[0, ...])
    rpt[2, ...] = np.arctan2(np.sqrt(xy), xyz[2, ...])
    return rpt


def rpt_to_xyz(rpt):
    """Convert from spherical to cartesian coordinates.

    Spherical is defined as ``(rad, phi, theta)``, where `phi` is the azimuthal angle (xy),
    and `theta` is the polar angle, with the +z-axis corresponding to ``theta = 0``.
    """
    rpt = np.asarray(rpt)
    if (np.ndim(rpt) < 1) or (np.shape(rpt)[0] != 3):
        raise ValueError("`rpt` ({}) must have shape (3, ...)!".format(rpt.shape))

    xyz = np.zeros_like(rpt)
    xyz[2, ...] = rpt[0, ...] * np.cos(rpt[2, ...])
    rxy = rpt[0, ...] * np.sin(rpt[2, ...])
    xyz[0, ...] = rxy * np.cos(rpt[1, ...])
    xyz[1, ...] = rxy * np.sin(rpt[1, ...])
    return xyz


def rtp_to_xyz(rtp):
    rtp = np.asarray(rtp)
    rpt = np.zeros_like(rtp)
    rpt[0, ...] = rtp[0, ...]
    rpt[1, ...] = rtp[2, ...]
    rpt[2, ...] = rtp[1, ...]
    return rpt_to_xyz(rpt)


def xyz_to_rtp(xyz):
    xyz = np.asarray(xyz)
    rpt = xyz_to_rpt(xyz)
    rtp = np.zeros_like(rpt)
    rtp[0, ...] = rpt[0, ...]
    rtp[1, ...] = rpt[2, ...]
    rtp[2, ...] = rpt[1, ...]
    return rtp


def zenum(*arr):
    utils.dep_warn("zenum", newname="zenumerate")
    return zenumerate(*arr)


def zenumerate(*arr):
    zipped = zip(*arr)
    return enumerate(zipped)


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

    if (start is stop is step is None):
        raise RuntimeError("``start``,``stop``, or ``step`` required!")

    ndim = np.ndim(arr)
    if (ndim == 0):
        ndim = arr

    if (ndim > 1):
        #     Create an object to slice all elements of all dims
        cut = [slice(None)]*ndim
        #     Exclude the last element of the last dimension
        cut[axis] = slice(start, stop, step)
        cut = tuple(cut)
    else:
        if (axis != 0) and (axis != -1):
            raise RuntimeError("cannot slice nonexistent axis!")
        cut = slice(start, stop, step)

    return cut


def spacing(data, scale='log', num=None, dex=10, dex_plus=None,
            filter=None, integers=False, endpoint=True, **kwargs):
    """Create an evenly spaced array between extrema from the given data.

    Arguments
    ---------
    data : array_like of scalar
        Data from which to extract the extrema for bounds.
    scale : str
        Scaling for spacing, {'lin', 'log'}.
    num : int
        Number of points to produce.
    dex : int
        Number of points per decade (order of magnitude) to produce.
    dex_plus : int or None
        After finding the number of bins using `dex` per decade, add this number of bins.
        If `dex` is given and `dex_plus` is None, `dex_plus` is set to 1 if endpoint, else 0
    filter : str or `None`
        String specifying how to filter the input `data` relative to zero.
    integers : bool
        Create spacing with only integer (whole numbers).  NOTE: this ignores the `num` argument.
        If ``scale == 'log'``: these are Scientific-notation integers, e.g. [7, 8, 9, 10, 20, 30].
        If ``scale == 'lin'``: these are all integers, e.g. [7, 8, 9, 10, 11, 12 ... 29].
        NOTE: when `integers` is 'True', the extrema are the nearest integral values *outside* the
              the range specified with `data`.  e.g. if `data` is [7.96, 28.12], the above example
              arrays are the ones that would be returned.
    **kwargs : dict arguments
        Additional arguments are passed to `minmax`, e.g. `log_stretch=0.1`.

    Returns
    -------
    spaced : (N,) array of scalar
        Array of evenly spaced points, with number of elements ``N = num`` if ``integers = False``,
        otherwise `N` is how many whole numbers there are between the given extrema (see `integers`)
        argument above.

    """
    DEF_NUM_LIN = 20

    if scale.startswith('log'):
        log_flag = True
    elif scale.startswith('lin'):
        log_flag = False
    else:
        raise RuntimeError("``scale`` '%s' unrecognized!" % (scale))

    # If no `filter` is given, and we are log-scaling, use a ``> 0.0`` filter
    if filter is None and log_flag:
        filter = '>'

    # Find extrema of values
    round = None
    # If only 'integers' (whole numbers) are desired, round extrema to *outside*
    if integers:
        round = 0
    span = minmax(data, filter=filter, round=round, round_scale=scale, **kwargs)
    if span is None:
        raise ValueError("Invalid span on input data!")

    if (num is None) and (not integers):
        if log_flag and (not integers):
            if dex_plus is None:
                dex_plus = 1 if endpoint else 0

            num_dex = np.fabs(np.diff(np.log10(span)))
            num = np.int(np.ceil(num_dex * dex)) + dex_plus
        else:
            num = DEF_NUM_LIN

    # If only 'integers', use 'arange'
    if integers:
        if num is not None:
            raise ValueError("`num` does not apply when using `integers` scaling!")

        # Log-spacing : create each decade manually
        if log_flag:
            # Find the SciNot exp for lower and upper values
            span_exp = np.floor(np.log10(span))
            # Find the SciNot mantissa for lower and upper values
            span_man = span / np.power(10, span_exp)
            # If range is only a single decade, create it directly
            if np.isclose(span_exp[0], span_exp[1]):
                spaced = np.arange(span_man[0], span_man[1]+1) * np.power(10.0, span_exp[0])
            # If multiple decades, create each separately
            else:
                exp_range = np.arange(span_exp[0], span_exp[1]+1)
                for ii, exp in enumerate(exp_range):
                    # If first decade, start at the first mantissa value
                    if ii == 0:
                        spaced = np.arange(span_man[0], 10) * np.power(10.0, exp)
                    # If last decade, end at the last mantissa value
                    elif ii == exp_range.size - 1:
                        spaced = np.append(spaced, np.arange(1, span_man[1]+1) * np.power(10.0, exp))
                    # Inbetween decades, use full range [1, 9]
                    else:
                        spaced = np.append(spaced, np.arange(1, 10) * np.power(10.0, exp))
        # Linear spacing
        else:
            spaced = np.arange(span[0], span[1]+1)

    # Create spacing between min/max values exactly
    else:
        if log_flag:
            spaced = np.logspace(*np.log10(span), num=num, endpoint=endpoint)
        else:
            spaced = np.linspace(*span, num=num, endpoint=endpoint)

    return spaced


def spacing_composite(comp_edges, scale, dex=None, num=None, **kwargs):
    if np.isscalar(scale):
        nsegs = len(comp_edges) - 1
        scale = [scale for ii in range(nsegs)]
    else:
        nsegs = len(scale)

    if len(comp_edges) != nsegs + 1:
        raise ValueError("`comp_edges` must have one more entry than `scale`")

    if dex is None and num is None:
        raise ValueError("`dex` or `num` must be provided!")

    if dex is None:
        dex = [None for ii in range(nsegs)]
    elif np.isscalar(dex):
        dex = [dex for ii in range(nsegs)]
    elif len(dex) != nsegs:
        raise ValueError("Length mismatch between `scale` and `dex`!")

    if num is None:
        num = [None for ii in range(nsegs)]
    elif np.isscalar(num):
        num = [num for ii in range(nsegs)]
    elif len(num) != nsegs:
        raise ValueError("Length mismatch between `scale` and `num`!")

    edges = []
    for ii, (sc, dd, nn) in enumerate(zip(scale, dex, num)):
        lo = comp_edges[ii]
        hi = comp_edges[ii+1]

        ep = True if (ii == nsegs - 1) else False
        dp = 1 if ep else 0

        tt = spacing([lo, hi], dex=dd, num=nn, endpoint=ep, dex_plus=dp, **kwargs)
        edges.append(tt)

    edges = np.concatenate(edges)

    return edges


def array_str(*args, **kwargs):
    return str_array(*args, **kwargs)


def str_array(arr, sides=(3, 3), delim=", ", format=None, log=False, label_log=True):
    """Create a string representation of a numerical array.

    Arguments
    ---------
    arr : array_like scalars,
        Array to be converted to string.
    sides : (2,) int, int, or None
        Specification for how many elements of the input array to print.
        -   (2,) int: then the first and second value determine the number of elements at the
                      beginning and end of the input array `arr` to print.
        -   int: the number of elements at both the beginning and end to print.
        -   None: print all elements of the array
    delim : str,
        Character to delimit elements of string array.
    format : str,
        Specification of how each array element should be converted to a str, e.g. (':.2f')
        This is a c-style specification used by ``str.format``.
    log : bool
        If this is True, first take the log10 of the input values before printing.
    label_log : bool
        If `log` is also true, append a string saying these are log values.

    Returns
    -------
    arr_str : str,
        Stringified version of input array.

    """
    arr = np.asarray(arr)
    if np.ndim(arr) > 1:
        arr = arr.flatten()

    if log:
        arr = np.log10(arr)

    len_arr = arr.size
    beg, end = _str_array_get_beg_end(sides, len_arr)

    if format is None:
        format = _guess_str_format_from_range(arr)

    # Create the style specification
    form = "{{{}}}".format(format)

    arr_str = _str_array_1d(arr, beg, end, form, delim)
    if log and label_log:
        arr_str += " (log values)"

    return arr_str


def str_array_neighbors(arr, inds, num=1, **kwargs):
    if np.ndim(arr) != 1 or np.ndim(inds) != 1:
        raise ValueError("`arr` and `inds` must be 1D!")

    if isinstance(inds[0], (bool, np.bool_)):
        inds = np.where(inds)[0]

    vals = []
    size = np.size(arr)
    for ii in inds:
        cut = slice(max(0, ii-num), min(size-1, ii+num+1))
        temp = str_array(arr[cut], sides=num+1, **kwargs)
        vals.append(temp)

    rv = "...".join(vals)
    return rv


def _guess_str_format_from_range(arr, prec=2, log_limit=2, allow_int=True):
    """
    """

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            extr = np.log10(np.fabs(minmax(arr, filter='ne')))

    # string values will raise a `TypeError` exception
    except (TypeError, AttributeError):
        return ":"

    if any(extr < -log_limit) or any(extr > log_limit):
        form = ":.{precision:d}e"
    elif np.issubdtype(arr.dtype, np.integer) and allow_int:
        form = ":d"
    else:
        form = ":.{precision:d}f"

    form = form.format(precision=prec)

    return form


def str_array_2d(arr, sides=(3, 3), delim=", ", format=None, log=False, label_log=True):
    arr = np.asarray(arr)
    assert np.ndim(arr) == 2, "Only supported for dim 2 arrays!"

    _def_format_small = ":7.2f"
    _def_format_large = ":7.2e"
    if (format is None):
        vmin, vmax = minmax(np.fabs(arr), filter='>')
        if (vmax >= 1e4) or (vmin < 1e-3):
            format = _def_format_large
        else:
            format = _def_format_small

    if log:
        arr = np.log10(arr)

    nrow, ncol = arr.shape
    rbeg, rend = _str_array_get_beg_end(sides, nrow)
    cbeg, cend = _str_array_get_beg_end(sides, ncol)

    # Create the style specification
    form = "{{{}}}".format(format)

    arr_str = []
    if rbeg is not None:
        for ii in range(rbeg):
            _str = _str_array_1d(arr[ii, :], cbeg, cend, form, delim)
            arr_str.append(_str)

    if (rbeg is not None) and (rend < nrow):
        arr_str.append("... ")

    for ii in range(rend):
        _str = _str_array_1d(arr[nrow-rend+ii, :], cbeg, cend, form, delim)
        arr_str.append(_str)

    arr_str = "\n".join(arr_str)
    if log and label_log:
        arr_str += " (log values)"

    return arr_str


def _str_array_1d(arr, beg, end, form, delim):
    arr_str = "["
    len_arr = arr.size

    # Add the first `first` elements
    if beg is not None:
        temp = [form.format(vv) for vv in arr[:beg]]
        # isinf = [tt in ['nan', 'inf', '-inf'] for tt in temp]
        # if np.any(isinf) and not np.all(isinf):
        #     size = len(temp[argfirst(~np.array(isinf))])
        #     fmt = "{{:^{len:}}}".format(len=size)

        arr_str += delim.join(temp)

    # Include separator unless full array is being printed
    if (beg is not None) or (end < len_arr):
        arr_str += "... "

    # Add the last `last` elements
    if end is not None:
        arr_str += delim.join([form.format(vv) for vv in arr[-end:]])

    arr_str += "]"
    return arr_str


def _str_array_get_beg_end(sides, size):
    if sides is None:
        beg = None
        end = size
    elif np.iterable(sides):
        beg, end = sides
    else:
        beg = end = sides

    _beg = 0 if beg is None else beg
    _end = 0 if end is None else end
    if _beg + _end >= size:
        beg = None
        end = size

    return beg, end


# def unify_shapes(*args):
#
#

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


def within(vals, extr, edges=True, all=False, inv=False, close=None):
    """Test whether a value or array is within the bounds of another.

    Arguments
    ---------
       vals  <scalar>([N])  : test value(s)
       extr  <scalar>[M]    : array or span to compare with
       edges <bool>         : optional, include the boundaries of ``extr`` as 'within'
       all   <bool>         : optional, whether All values are within bounds (single return `bool`)
       inv   <bool>         : optional, invert results (i.e. `True` ==> `False` and visa-versa)

    Returns
    -------
       <bool> : True if within, False otherwise

    """

    if np.ndim(extr) != 1:
        raise ValueError("`extr` must be 1D!")

    vals = np.asarray(vals)

    nex = np.size(extr)
    if nex < 2:
        raise ValueError("`extr` must have at least 2 elements!")
    elif nex == 2:
        extr_bnds = [ex for ex in extr]
        extr_bnds[0] = -np.inf if (extr_bnds[0] is None) else extr_bnds[0]
        extr_bnds[1] = +np.inf if (extr_bnds[1] is None) else extr_bnds[1]
        if np.diff(extr_bnds) <= 0.0:
            err = "Extrema values are not ordered!  '{}' ==> '{}'".format(extr, extr_bnds)
            raise ValueError(err)
    else:
        extr_bnds = minmax(extr)

    # Include edges for WITHIN bounds (thus not including is outside)
    if edges:
        above = (vals >= extr_bnds[0])
        below = (vals <= extr_bnds[1])
    else:
        above = (vals > extr_bnds[0])
        below = (vals < extr_bnds[1])

    if close is not None:
        if close is True:
            close = dict()
        above = above | np.isclose(vals, extr_bnds[0], **close)
        below = below | np.isclose(vals, extr_bnds[1], **close)

    retval = np.asarray(above & below)

    # Convert to single return value
    if all:
        retval = np.all(retval)

    # Invert results
    if inv:
        retval = np.invert(retval)

    return retval


def _comparison_function(comp, value=0.0, **kwargs):
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
        return func(xx, value, **kwargs)

    return comp_func


def comparison_filter(data, filter, inds=False, value=0.0, finite=True, mask=False, **kwargs):
    """
    """
    if filter is None:
        return data
    if not callable(filter):
        filter = _comparison_function(filter, value=value, **kwargs)

    # Include is-finite check
    if finite:
        sel = filter(data) & np.isfinite(data)
    else:
        sel = filter(data)

    if inds:
        return sel
    else:
        if mask:
            return np.ma.masked_where(~sel, data)
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
    if isinstance(frac, numbers.Integral):
        return frac

    if (round == 'floor'):
        roundFunc = np.floor
    elif (round == 'ceil'):
        roundFunc = np.ceil
    else:
        raise ValueError("Unrecognized ``round``!")

    # Convert fractional input into an integer
    if (within is not None):
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
