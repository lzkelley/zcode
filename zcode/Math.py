"""
Functions for Math operations.

Functions
---------
 - spline                               : Create a general spline interpolation function.
 - logSpline_resample                   : use a log-log spline to resample the given data
 - logSpline                            : construct a spline in log-log space
 - logSpline_mono                       : monotonic, cubic spline interpolant in log-log-space.
 - contiguousInds                       : find the largest segment of contiguous array values
# - cumtrapz_rev
 - cumtrapz_loglog
 - within
 - minmax                               : find the min and max of given values
 - spacing                              : Create an even spacing between extrema from given data.
 - histogram                            : performed advanced binning operations
 - sliceForAxis                         : Array slicing object which slices only the target axis.
 - midpoints                            : Return the midpoints between values in the given array.
 - vecmag                               : find the magnitude/distance of/between vectors
 - extend                               : Extend the given array by extraplation.
 - renumerate                           : construct a reverse enumeration iterator
 - cumstats
 - confidenceIntervals
 - frexp10                              : decompose a float into mantissa and exponent (base 10)
 - stats                                : return ave, stdev
 - groupDigitized                       : Find groups of array indices corresponding to each bin.
 - sampleInverse                        : Find x-sampling to evenly divide a function in y-space.

 - smooth

# createSlice

"""

import itertools
import numpy as np
import scipy as sp
import scipy.interpolate
import warnings, numbers


def spline(xx, yy, order=3, log=True, mono=False, extrap=True, pos=False):
    """
    Create a general spline interpolation function.

    Arguments
    ---------
        xx     <flt>[N] : Input independent variable
        yy     <flt>[N] : Input   dependent variable
        order  <int>    : order of interpolation (must be ``3`` if ``mono``)
        log    <bool>   : interpolate in log-log-space
        mono   <bool>   : use a specifically monotonic interpolator
        extrap <bool>   : allow extrapolation outside of range of ``xx``
        pos    <bool>   : filter to only positive values of ``yy``

    Returns
    -------
        spline <obj>    : callable function returning the interpolated values.

    """

    xp = np.array(xx)
    yp = np.array(yy)

    # Select positive y-values
    if( pos ):
        inds = np.where( yp > 0.0 )[0]
        xp = xp[inds]
        yp = yp[inds]

    # Convert to log-space as needed
    if( log ):
        xp = np.log10(xp)
        yp = np.log10(yp)

    # Sort input arrays
    inds = np.argsort(xp)
    xp = xp[inds]
    yp = yp[inds]

    # Monotonic Interpolation
    if( mono ):
        if( order != 3 ): warnings.warn("monotonic `PchipInterpolator` is always cubic!")
        terp = sp.interpolate.PchipInterpolator(xp, yp, extrapolate=extrap)
    # General Interpolation
    else:
        # Let function extrapolate outside range
        if( extrap ): ext = 0
        # Return zero outside of range
        else:         ext = 1
        terp = sp.interpolate.InterpolatedUnivariateSpline(xp, yp, k=order, ext=ext)


    # Convert back to normal space, as needed
    if( log ): spline = lambda xx: np.power(10.0, terp(np.log10(xx)))
    else:      spline = terp

    return spline

# } spline()


def logSpline_resample(xx, yy, newx, order=3):
    """
    Use a log-spline to resample the given function at new points.

    Arguments
    ---------
       xx   : <scalar>[N], independent variable of original function
       yy   : <scalar>[N], dependent variable of original function
       newx : <scalar>[M], new independent variable points at which to resample

    Returns
    -------
       newy : <scalar>[M], resampled function values

    """
    spliner = logSpline(xx, yy, order=order)
    newy    = spliner(newx)
    return newy


def logSpline(xx, yy, order=3, pos=True):
    """
    Create a spline interpolant in log-log-space.

    Extrapolates to new values outside of range

    Parameters
    ----------
        xx : array, independent variable
        yy : array, function of ``xx``
        order : int, order of spline interpolant

    Returns
    -------
        spline : callable, spline interpolation function

    """

    if( pos ):
        inds = np.where( (xx > 0.0) & (yy > 0.0) )
        xl = np.log10(xx[inds])
        yl = np.log10(yy[inds])
        if( len(inds[0]) < order+1 ):
            raise RuntimeError("Too few valid valies (%d) for order %d!!" % (len(inds), order))

    else:
        xl = np.log10(xx)
        yl = np.log10(yy)


    terp = sp.interpolate.InterpolatedUnivariateSpline(xl, yl, k=order)
    spline = lambda xx: np.power(10.0, terp(np.log10(xx)))

    return spline

# } logSpline()



def logSpline_mono(xx, yy, pos=True, extrap=False):
    """
    Create a *monotonic*, cubic spline interpolant in log-log-space.

    Parameters
    ----------
        xx     <flt>[N] : independent variable
        yy     <flt>[N] : function of ``xx``
        pos    <bool>   : select only positive values from ``yy`` (and corresponding ``xx``)
        extrap <bool>   : allow extrapolation outside of bounds of ``xx``

    Returns
    -------
        spline <call>   : spline interpolation function.

    """

    order = 2

    if( pos ):
        inds = np.where( (xx > 0.0) & (yy > 0.0) )
        xl = np.log10(xx[inds])
        yl = np.log10(yy[inds])
        if( len(inds[0]) < order+1 ):
            raise RuntimeError("Too few valid valies (%d) for order %d!!" % (len(inds), order))

    else:
        xl = np.log10(xx)
        yl = np.log10(yy)

    terp = sp.interpolate.PchipInterpolator(xl, yl, extrapolate=extrap)
    spline = lambda xx: np.power(10.0, terp(np.log10(xx)))

    return spline

# } logSpline_mono()



def contiguousInds(args):
    """
    Find the longest contiguous segment of positive values in the array.
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
    idx.shape = (-1,2)

    # Find lengths of each contiguous segment
    sizes = np.diff( idx, axis=1 )
    # Find the location of maximum segment length
    maxPos = np.argmax(sizes)
    # Make indices spanning longest segment
    inds = np.arange(*idx[maxPos])

    return inds

# contiguousInds()

'''
def cumtrapz_rev(yy, xx=None, initial=0.0):
    """
    """
    if( xx is not None ):
        assert np.shape(yy) == np.shape(xx), "Shapes of ``yy`` and ``xx`` must match!"

    # Reverse the last axis
    cut = sliceForAxis(yy, axis=-1, step=-1)

    yy = yy[cut]
    if( xx is not None ): xx = xx[cut]

    integ = sp.integrate.cumtrapz(yy, x=xx, initial=initial)[cut]

    return integ

# } cumtrapz_rev()
'''


def cumtrapz_loglog(yy, xx, init=0.0, rev=False):
    """
    From Thomas Robitaille
    https://github.com/astrofrog/fortranlib/blob/master/src/lib_array.f90
    """

    if( np.ndim(yy) > 1 ): raise RuntimeError("This isn't implemented for ndim > 1!")

    nums = len(xx)
    sum = np.zeros(nums)

    if( rev ): 
        xx = xx[::-1]
        yy = yy[::-1]

    sum[0] = init
    for ii in xrange(1,nums):
        sum[ii] = sum[ii-1] + _trapezium_loglog(xx[ii-1], yy[ii-1], xx[ii], yy[ii])

    if( rev ): sum = sum[::-1]

    return sum


def _trapezium_loglog(x1,y1,x2,y2):
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
    """
    Test whether a value or array is within the bounds of another.

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
    if( edges ): retval = np.asarray( ((vals >= extr_bnds[0]) & (vals <= extr_bnds[1])) )
    # Don't include edges for WITHIN  (thus include them for outside)
    else:        retval = np.asarray( ((vals >  extr_bnds[0]) & (vals <  extr_bnds[1])) )

    # Convert to single return value
    if( all ): retval = np.all(retval)

    # Invert results
    if( inv ): retval = np.invert(retval)

    return retval

# within()



def minmax(data, nonzero=False, positive=False, prev=None, stretch=0.0):
    """
    Find minimum and maximum of given data, return as numpy array.

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
    if( nonzero ):  useData = np.array(useData[np.nonzero(useData)])
    if( positive ): useData = np.array(useData[np.where(useData >= 0.0)])

    # If there are no elements (left), return ``prev`` (`None` if not provided)
    if( np.size(useData) == 0 ): return prev

    # Determine stretch factor
    lef = (1.0-stretch)
    rit = (1.0+stretch)

    # Find extrema
    minmax = np.array([lef*np.min(useData), rit*np.max(useData)])

    # Compare to previous extrema, if given
    if( prev is not None ):
        minmax[0] = np.min([minmax[0],prev[0]])
        minmax[1] = np.max([minmax[1],prev[1]])

    return minmax

# minmax()



def spacing(data, scale='log', num=100, nonzero=None, positive=None):
    """
    Create an evenly spaced array between extrema from the given data.

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

    if(   scale.startswith('log') ): log_flag = True
    elif( scale.startswith('lin') ): log_flag = False
    else: raise RuntimeError("``scale`` '%s' unrecognized!" % (scale))

    if( nonzero is None ):
        if( log_flag ): nonzero = True
        else:           nonzero = False

    if( positive is None ):
        if( log_flag ): positive = True
        else:           positive = False

    span = minmax(data, nonzero=nonzero, positive=positive)
    if(   log_flag ): spacing = np.logspace( *np.log10(span), num=num )
    else:             spacing = np.linspace( *span,           num=num )

    return spacing

# spacing()



def histogram(args, bins, binScale=None, bounds='both',
              weights=None, func='sum', cumul=False, stdev=False):
    """
    Histogram (bin) the given values.

    - Currently ``bins`` must be monotonically increasing!!
    - When using bounds=='both', you currently can't control which interior bin edges are inclusive

    Arguments
    ---------
       args     <scalar>[N]   : data to be histogrammed.
       bins     <scalar>([M]) : Positions of bin edges or number of bins to construct automatically.
                                If a single ``bins`` value is given, it is assumed to be a number of
                                bins to create with a scaling given by ``binScale`` (see desc).
       binScale <str>         : scaling to use when creating bins automatically.
                                If `None`, the extrema of ``args`` are used to make an selection.
       bounds    <str>        : optional, how to treat the given ``bins`` values, i.e. which 'edges'
                                these represent.  Must be one of {`left`, `both`, `right`}.
                                ``left``  : then ``bins[0]`` is the left-most inclusive edge
                                            (values beyong that will not be counted), and there is
                                            no right-most edge
                                ``right`` : then ``bins[-1]`` is the right-most inclusive edge
                                            (values beyond that will not be counted), and there is
                                            no  left-most edge
                                ``both``  : then values outside of the range of ``bins`` will not be
                                            counted anywhere.  Returned histogram will have length
                                            `M-1` --- representing space between ``bins`` values
       weights  <scalar>([N]) : optional, weighting factors for each input value in ``args``
       func     <str>         : optional, what binning operation to perform; in each bin:
                                ``sum``   : sum all values (weights, or count)
                                ``ave``   : average      of ``weights``
                                ``max``   : find maximum of ``weights``
                                ``min``   : find minimum of ``weights``
       cumul    <bool>        : also calculate and return a cumulative distribution of ``counts``
       stdev    <bool>        : optional, find standard-deviation of ``weights`` in each bin


    Returns
    -------
       edges    <scalar>[L]   : edges used for creating histogram
       counts   <int>[L]      : histogram of counts per bin, if ``edges`` is `both` then length is
                               `L=M-1`, otherwise `L=M`.

       cumsum   <int>[L]      : optional, cumulative distribution of ``counts``,
                                returned if ``cumul`` is True
       hist     <scalar>[L]   : optional, histogram of ``func`` operation on ``weights``
                                returned if ``weights`` is given.
       std      <scalar>[L]   : optional, standard-deviation of ``weights`` in bin,
                                returned if ``stdev == True``


    To-Do
    -----
     - Allow multiple ``funcs`` to be performed simultaneously, i.e. 'min' and 'max'
     - Changes ``bounds`` options to be 'inner', 'outer', 'left', 'right'

    """

    assert func in [ 'sum', 'ave', 'min', 'max' ], "Invalid ``func`` argument!"

    # For anything besides counting ('sum'), we need a weight for each argument
    if( func is not 'sum' or stdev == True ):
        assert np.shape(weights) == np.shape(args), "Shape of ``weights`` must match ``args``!"


    ## Prepare Effective bin edges as needed
    #  -------------------------------------

    rightInclusive = False

    # Construct a certain number ``bins`` bins spaced appropriately
    if( np.size(bins) == 1 ):
        extr = minmax(args)
        useScale = binScale
        # If no bin scaling is given, make an appropriate choice
        if( useScale is None ):
            useScale = 'log'
            # Dont use log if there are zero/negative values
            if(   extr[0] <= 0.0 ):         useScale = 'lin'
            # Dont use log for small ranges of values
            elif( extr[1]/extr[0] < 10.0 ): useScale = 'lin'

        edges = spacing(extr, scale=useScale, num=bins+1, nonzero=False)
    else:
        edges = np.array(bins)


    # Design a right-most bin to include right-most particles
    if(   bounds == 'left'  ):
        # Try to go just after right-most value
        useMax  = 1.01*np.max([np.max(edges), np.max(args)])
        # Deal with right-most being 0.0
        shiftMax = minmax(np.fabs(args), nonzero=True)
        #     If there are no, nonzero values ``None`` is returned
        if( shiftMax is None ): shiftMax = 1.0
        else:                   shiftMax = 0.1*shiftMax[0]
        useMax += shiftMax
        edges = np.concatenate([edges, [useMax]])
    # Design a left-most  bin to include left-most  particles
    elif( bounds == 'right' ):
        # Try to go just before left-most value
        useMin  = 0.99*np.min([np.min(edges), np.min(args)])
        # Deal with left-most being 0.0
        shiftMin = minmax(np.fabs(args), nonzero=True)
        #     If there are no, nonzero values ``None`` is returned
        if( shiftMin is None ): shiftMin = 1.0
        else:                   shiftMin = 0.1*shiftMin[0]
        useMin -= shiftMin
        edges = np.concatenate([[useMin], edges])
        rightInclusive = True
    elif( bounds == 'both'  ):
        pass
    else:
        raise RuntimeError("Unrecognized ``bounds`` parameter!!")


    ## Find bins for each value
    #  ------------------------

    # Find where each value belongs
    digits = np.digitize(args, edges, right=rightInclusive)
    # Histogram values (i.e. count entries in bins)
    counts = [ np.count_nonzero(digits == ii) for ii in range(1, len(edges)) ]
    # Add values equaling the right-most edge
    if( bounds == 'both' ): counts[-1] += np.count_nonzero( args == edges[-1] )

    counts = np.array(counts)

    # Calculate cumulative distribution
    if( cumul ): cumsum = np.cumsum(counts)

    # Just histogramming counts
    if( weights is None ):
        if( cumul ): return edges, counts, cumsum
        else:        return edges, counts


    ## Perform Weighting
    #  -----------------

    weights = np.array(weights)

    # if a single scaling is provided
    if( np.size(weights) == 1 ):
        if( cumul ): return edges, counts, cumsum, counts*weights
        else:        return edges, counts, counts*weights


    # If ``weights`` has values for each argument ``args``

    useFunc = np.sum
    if(   func == 'min' ): useFunc = np.min
    elif( func == 'max' ): useFunc = np.max

    # Sum values in bins
    hist = [ useFunc(weights[digits == ii]) for ii in range(1, len(edges)) ]
    # Add values equaling the right-most edge
    if( bounds == 'both' ):
        # Add values directly into right-most bin
        if( func == 'ave' or func == 'sum' ):
            hist[-1] += useFunc( weights[args == edges[-1]] )
        # Find min/max of values compared to whats already in right-most bin
        else:
            hist[-1]  = useFunc( [hist[-1],weights[args == edges[-1]]] )

    # Average Bins
    if( func == 'ave' ): hist = [ vv/hh if hh > 0 else 0.0 for hh,vv in zip(counts, hist) ]

    hist = np.array(hist)


    # Calculate standard-deviations
    # -----------------------------

    if( stdev ):

        # Sum values in bins
        std = [ np.std(weights[digits == ii]) for ii in range(1, len(edges)) ]
        # Fix last bin to include values which equal the right-most edge
        if( bounds == 'both' ):
            std[-1] = np.std( weights[ (digits == ii) | (args == edges[-1]) ] )

        std = np.array(std)

        if( cumul ): return edges, counts, hist, std
        else:        return edges, counts, cumsum, hist, std


    # No ``std``, just return histograms of counts and ``func`` on ``weights``
    if( cumul ): return edges, counts, cumsum, hist
    else:        return edges, counts, hist

# histogram()


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

    if( start == stop == step == None ): 
        raise RuntimeError("``start``,``stop``, or ``step`` required!")

    ndim = np.ndim(arr)
    if( ndim == 0 ): ndim = arr

    if( ndim > 1 ):
        #     Create an object to slice all elements of all dims
        cut = [slice(None)]*ndim
        #     Exclude the last element of the last dimension
        cut[axis] = slice(start,stop,step)
    else:
        if( axis != 0 and axis != -1 ): raise RuntimeError("cannot slice nonexistent axis!")
        cut = slice(start, stop, step)

    return cut

# } sliceForAxis()    


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

    if( np.shape(arr)[-1] < 2 ):
        raise RuntimeError("Input ``arr`` does not have a valid shape!")

    if( log ): user = np.log10(arr)
    else:      user = np.array(arr)

    diff = np.diff(user)

    # skip the last element, or the last axis
    cut = sliceForAxis(arr, axis=-1, stop=-1)
    start = arr[cut]
    mids = start + frac*diff

    if( log ): mids = np.power(10.0, mids)

    return mids

# } midpoints()



def vecmag(r1, r2=None):
    """
    Calculate the distance from vector(s) r1 to r2.

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

    if( r2 is None ): r2 = np.zeros(np.shape(r1))

    if( len(np.shape(r1)) > 1 or len(np.shape(r2)) > 1 ):
        dist = np.sqrt( np.sum( np.square(r1 - r2), axis=1) )
    else:
        dist = np.sqrt( np.sum( np.square(r1 - r2) ) )

    return dist

# vecmag()


def extend(arr, num=1, log=True, append=False, side='both'):
    """
    Extend the given array by extraplation.

    Arguments
    ---------
        arr    <flt>[N] : array to extend
        num    <int>    : number of points to add (on each side, if ``both``)
        log    <bool>   : extrapolate in log-space
        append <bool>   : add the extended points onto the given array
        side   <str>    : which side to extend from {'left','both','right'}.

    Returns
    -------
        retval <flt>[M] : extension (or input ``arr`` with extension added, if ``append``).

    """

    if( log ): useArr = np.log10(arr)
    else:      useArr = np.array(arr)

    steps = np.arange(1,num+1).squeeze()
    if( side == 'both' or side == 'left' ): left = useArr[ 0] + (useArr[ 0] - useArr[ 1])*steps
    else: left = []
    if( side == 'both' or side == 'right'): rigt = useArr[-1] + (useArr[-1] - useArr[-2])*steps
    else: rigt = []

    if( log ):
        left = np.power(10.0, left)
        rigt = np.power(10.0, rigt)

    if( append ): retval = np.hstack([left, arr, rigt])
    else:         retval = np.hstack([left, rigt])

    return retval

# extend()


def renumerate(arr):
    """
    Same as ``enumerate`` but in reverse order.  Uses iterators, no copies made.
    """
    return itertools.izip(reversed(xrange(len(arr))), reversed(arr))

# renumerate()


def cumstats(arr):
    """
    Calculate a cumulative average and standard deviation.
    """

    tot = len(arr)
    num = np.arange(tot)
    std = np.zeros(tot)
    # Cumulative sum
    sm1 = np.cumsum(arr)
    # Cumulative sum of squares
    sm2 = np.cumsum( np.square(arr) )
    # Cumulative average
    ave = sm1/(num+1.0)

    std[1:] = np.fabs(sm2[1:] - np.square(sm1[1:])/(num[1:]+1.0))/num[1:]
    std[1:] = np.sqrt( std[1:] )
    return ave,std

# cumstats()



def confidenceIntervals(vals, ci=[0.68, 0.95, 0.997]):
    """
    Compute the values bounding the target confidence intervals for an array of data.

    Arguments
    ---------
       vals <scalar>[N] : array of sample data points
       ci   <scalar>[M] : optional, list of confidence intervals as fractions (e.g. `[0.68, 0.95]`)

    Returns
    -------
       med  <scalar>      : median of the data
       conf <scalar>[M,2] : bounds for each confidence interval

    """

    if( not np.iterable(ci) ): ci = [ ci ]
    ci = np.asarray(ci)
    assert np.all(ci >= 0.0) and np.all(ci <= 1.0), "Confidence intervals must be {0.0,1.0}!"

    cdf_vals = np.array([(1.0-ci)/2.0, (1.0+ci)/2.0 ]).T
    conf = [ [np.percentile(vals, 100.0*cdf[0]), np.percentile(vals, 100.0*cdf[1])]
             for cdf in cdf_vals ]
    conf = np.array(conf)
    med = np.percentile(vals, 50.0)

    if( len(conf) == 1 ): conf = conf[0]

    return med, conf

# confidenceIntervals()


def frexp10(vals):
    """
    Return the mantissa and exponent in base 10

    Arguments
    ---------
        vals <flt>(N) : values to be converted

    Returns
    -------
        man <flt>(N) : mantissa
        exp <flt>(N) : exponent

    """

    exp = np.int( np.floor(np.log10(vals)) )
    man = vals / np.power(10.0, exp)
    return man, exp

# frexp10()


def stats(vals):
    """
    """
    ave = np.average(vals)
    std = np.std(vals)
    return ave, std

# } stats()



def groupDigitized(arr, bins, edges='right'):
    """
    Find groups of array indices corresponding to each bin.

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

    if(   edges.startswith('r') ): right = True
    elif( edges.startswith('l') ): right = False
    else: RuntimeError("``edges`` must be 'right' or 'left'!")

    # ``numpy.digitize`` always assumes ``bins`` are right-edges (in effect)
    shift = 0
    # If we want 'left' bin edges, such shift each bin leftwards
    if( not right ): shift = -1

    # Find in which bin each element of arr belongs
    pos = np.digitize(arr, bins, right=right) + shift

    groups = []
    # Group indices by bin number
    for ii in xrange(len(bins)):
        groups.append( np.where(pos == ii)[0] )

    return groups

# } groupDigitized()


def sampleInverse(xx, yy, num=100, log=True, sort=True):
    """
    Find the x-sampling of a function to evenly divide its results in y-space.

    Input function *must* be strictly monotonic in ``yy``.

    Arguments
    ---------
        xx   <flt>[N] : array(scalar), initial sample space
        yy   <flt>[N] : function to resample
        num  <int>    : number of points to produce
        log  <bool>   : sample in log space
        sort <bool>   : sort ``yy``; if ``False``, ``yy`` must be sorted already

    Returns
    -------
        samps <flt>[``num``] : new sample points from ``xx``

    """

    # Convert to log-space, as needed
    if( log ):
        xp = np.log10(xx)
        yp = np.log10(yy)
    else:
        xp = np.array(xx)
        yp = np.array(yy)

    # y-values must be sorted
    if( sort ):
        inds = np.argsort(yp)
        xp = xp[inds]
        yp = yp[inds]
    else:
        # Check for strict-monotonicity
        if( not all(left < right for left,right in zip(yp, yp[1:])) and
            not all(left > right for left,right in zip(yp, yp[1:])) ):
            raise RuntimeError("Input must be monotonic!")


    # Construct Interpolating Function, *must be monotonic*
    interpBack = spline(yp, xp, log=False, mono=True)

    # Divide y-axis evenly, and find corresponding x-points
    levels = spacing(yp, scale='lin', num=num)
    samples = interpBack(levels)

    # Convert back to normal space, as needed
    if( log ): samples = np.power(10.0, samples)

    return samples

# } sampleInverse()





def smooth(arr, size, width=None, loc=None, mode='same'):
    """
    Use convolution to smooth the given array.

    The ``width`` and ``loc`` arguments can be given as integers, in which case they are taken
    as indices in the input array; or they can be floats, in which case they are interpreted as
    fractions of the length of the input array.

    Arguments
    ---------
        arr   <flt>[N] : input array to be smoothed
        size  <int>    : size of smoothing window
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

    assert size <= length, "``size`` must be less than length of input array!"

    window = np.ones(int(size))/float(size)

    # Smooth entire array
    smArr = np.convolve(arr, window, mode=mode)

    # Return full smoothed array if no bounds given
    if( width is None ):
        return smArr

    # Other convolution modes require dealing with differing lengths
    #    If smoothing only a portion of the array, 
    assert mode == 'same', "Other convolution modes not supported for portions of array!"

    ## Smooth portion of array
    #  -----------------------

    if( np.size(width) == 2 ):
        lef = width[0]
        rit = width[1]
    elif( np.size(width) == 1 ):
        if( loc is None ): raise ValueError("For a singular ``width``, ``pos`` must be provided!")
        lef = width
        rit = width
    else:
        raise ValueError("``width`` must be one or two scalars!")


    # Convert fractions to positions, if needed
    lef = _fracToInt(lef, length-1, within=1.0, round='floor')
    rit = _fracToInt(rit, length-1, within=1.0, round='floor')
        
    # If ``loc`` is provided, use ``width`` relative to that
    if( loc is not None ):
        loc = _fracToInt(loc, length-1, within=1.0, round='floor')
        lef = loc - lef
        rit = loc + rit

    
    mask = np.ones(length, dtype=bool)
    mask[lef:rit] = False
    smArr[mask] = arr[mask]

    return smArr
# } smooth()


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
    if( isinstance(frac, numbers.Integral) ): return frac

    if(   round == 'floor' ):
        roundFunc = np.floor
    elif( round == 'ceil' ):
        roundFunc = np.ceil
    else:
        raise ValueError("Unrecognized ``round``!")

    # Convert fractional input into an integer
    if( within is not None ):
        assert frac >= 0.0 and frac <= within, "``frac`` must be between [0.0,%s]!" % (str(within))

    loc = np.int(roundFunc(frac*size))

    return loc

# } _fracToInt()


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
    if( isinstance(index, numbers.Integral) ):
        cut = slice(index, index+1)
        ids = np.arange(index, index+1)
    # Range of values
    elif( np.iterable(index) ):
        cut = index
        ids = index
    else:
        if( index is not None ):
            self.log.error("Unrecognized `index` = '%s'!" % (str(index)))
            self.log.warning("Returning all entries")

        cut = slice(None)
        ids = np.arange(max)


    return ids, cut

# } createSlice()
'''

