"""
Functions for Math operations.

Functions
---------
 - logSpline_resample()
 - logSpine()
 - contiguousInds()
 - integrate_cumulative_simpson()
 - integrate_cumulative_func_simpson()
 - integrate_cumulative_arr_trapezoid()
 - within()
 - minmax()
 - spacing()
 - histogram()
 - mid()
 - dist()
 - extend()
 - renumerate()
 - cumstats()
 - confidenceIntervals()

"""

import itertools
import numpy as np
import scipy as sp
import scipy.interpolate


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


def logSpline(xx, yy, order=3):
    """
    Create a spline interpolant in log-log-space.

    Parameters
    ----------
    xx : array, independent variable
    yy : array, function of ``xx``
    order : int, order of spline interpolant

    Returns
    -------
    spline : callable, spline interpolation function

    """

    xl = np.log10(xx)
    yl = np.log10(yy)
    terp = sp.interpolate.InterpolatedUnivariateSpline(xl, yl, k=order)
    spline = lambda xx: np.power(10.0, terp(np.log10(xx)))

    return spline


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





def integrate_cumulative_simpson(func, xx, log=True):
    """
    Perform a cumulative integral of a callable functon using Simpson's rule.

    Notes
    -----
      See: https://en.wikipedia.org/wiki/Simpson%27s_rule


    """

    left = xx[:-1]
    rigt = xx[1:]
    diff = np.diff(xx)
    if( log ): cent = np.power(10.0, 0.5*(np.log10(left) + np.log10(rigt)))
    else:      cent = 0.5*(left + rigt)

    retval = func(left) + 4.0*func(cent) + func(rigt)
    retval = np.cumsum((diff/6.0)*retval)
    return retval

# integrate_cumulative_simpson()



def integrate_cumulative_func_simpson(func, xx, log=True, init=None):
    """
    Perform a cumulative integral of a callable functon using Simpson's rule.

    Notes
    -----
      See: https://en.wikipedia.org/wiki/Simpson%27s_rule


    """

    left = xx[:-1]
    rigt = xx[1:]
    diff = np.diff(xx)
    if( log ): cent = np.power(10.0, 0.5*(np.log10(left) + np.log10(rigt)))
    else:      cent = 0.5*(left + rigt)

    retval = func(left) + 4.0*func(cent) + func(rigt)
    retval = np.cumsum((diff/6.0)*retval)


    ## If an initial value is provided, include in integral
    if( init is not None ):
        # Add to all values
        retval += init
        # Prepend as initial value (i.e. for ``xx[0]``)
        #if( rev ): retval = np.insert(retval, len(retval), init)
        #else:      retval = np.insert(retval, 0, init)
        retval = np.insert(retval, 0, init)


    return retval

# integrate_cumulative_simpson()



def integrate_cumulative_arr_trapezoid(arr, xx, log=True, init=None, rev=False):
    """
    Perform a cumulative integral of a array using the Trapezoidal Rule

    Arguments
    ---------
    arr  : <scalar>[N], array of values to be integrated
    xx   : <scalar>[N], array of spacings used as the variable of integration (i.e. independent)
    log  : (<bool>),    perform integral in log-space (i.e. average the array in log-space)
    init : (<scalar>),  initial value of integral (i.e. constant of integration), if this value is
                        provided (i.e. not-`None`) then it will be added to each element of the
                        integrated values and prepended to the returned array.

    Returns
    -------
    retval : <scalar>[*], integrated array, if ``init`` is NOT provided the length of the returned
                          array will be [N-1] --- i.e. the number of intervals in the array, but if
                          ``init`` IS provided, the ``init`` value will be prepended and the
                          returned length will be [N].

    Notes
    -----
      See: https://en.wikipedia.org/wiki/Trapezoidal_rule

    """

    ## Find left and right bin edges, and bin widths
    lx = xx[:-1]
    rx = xx[1:]
    diff = np.diff(xx)

    ## Find left and right array values, and averages
    ly = arr[:-1]
    ry = arr[1:]
    # Average in log-space if desired
    if( log ): aves = np.power(10.0, 0.5*(np.log10(ly) + np.log10(ry)))
    else:      aves = 0.5*(ly + ry)

    if( rev ): retval = np.cumsum(diff[::-1]*aves[::-1])[::-1]
    else:      retval = np.cumsum(diff*aves)

    ## If an initial value is provided, include in integral
    if( init is not None ):
        # Add to all values
        retval += init
        # Prepend as initial value (i.e. for ``xx[0]``)
        # retval = np.insert(retval, 0, init)
        if( rev ): retval = np.insert(retval, len(retval), init)
        else:      retval = np.insert(retval, 0, init)

    return retval

# integrate_cumulative_arr_trapezoid()



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



def minmax(data, nonzero=False, prev=None):
    """
    Find minimum and maximum of given data, return as numpy array.

    If ``prev`` is provided, the returned minmax values will also be compared to it.

    Arguments
    ---------
       data    <scalar>[...] : arbitrarily shaped data to find minumum and maximum of.
       prev    <scalar>[2]   : optional, also find min/max against prev[0] and prev[1] respectively.
       nonzero <bool>        : optional, ignore zero valued entries

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
    if( nonzero ): useData = np.array(useData[np.nonzero(useData)])

    # If there are no elements (left), return ``prev`` (`None` if not provided)
    if( np.size(useData) == 0 ): return prev

    # Find extrema
    minmax = np.array([np.min(useData), np.max(useData)])

    # Compare to previous extrema, if given
    if( prev is not None ):
        minmax[0] = np.min([minmax[0],prev[0]])
        minmax[1] = np.max([minmax[1],prev[1]])

    return minmax

# minmax()



def spacing(data, scale='log', num=100, nonzero=True, positive=False):
    """
    Create an evenly spaced array between extrema from the given data.

    Arguments
    ---------
       data     <scalar>[M] : data from which to extract the extrema for bounds
       scale    <str>       : optional, scaling for spacing, {'lin', 'log'}
       num      <int>       : optional, number of points, ``N``
       nonzero  <bool>      : optional, only use nonzero  elements of ``data``
       positive <bool>      : optional, only use positive elements of ``data``
    
    Returns
    -------
       spacing <scalar>[N] : array of evenly spaced points, with number of elements ``N = num``

    """


    usedata = np.array(data)
    if( nonzero  ): usedata = usedata[np.nonzero(usedata)]
    if( positive ): usedata = usedata[np.where(usedata > 0.0)]

    span = minmax(usedata)
    if(   scale.startswith('log') ): spacing = np.logspace( *np.log10(span), num=num )
    elif( scale.startswith('lin') ): spacing = np.linspace( span,            num=num )
    else: raise RuntimeError("``scale`` unrecognized!")

    return spacing

# spacing()



def histogram(args, bins, weights=None, func='sum', edges='both', stdev=False):
    """
    Histogram (bin) the given values.

    - Currently ``bins`` must be monotonically increasing!!
    - When using edges=='both', you currently can't control which interior bin edges are inclusive

    Arguments
    ---------
       args    <scalar>[N]   :
       bins    <scalar>[M]   : 
       weights <scalar>([N]) : optional, weighting factors for each input value in ``args``
       func    <str>         : optional, what binning operation to perform; in each bin:
                               ``sum``   : sum all values (weights, or count)
                               ``ave``   : average      of ``weights``
                               ``max``   : find maximum of ``weights``
                               ``min``   : find minimum of ``weights``
       edges   <str>         : optional, how to treat the given ``bins`` values, i.e. which 'edges'
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
       stdev   <bool>        : optional, find standard-deviation of ``weights`` in each bin

    Returns
    -------
       counts  <int>[L]      : histogram of counts per bin, if ``edges`` is `both` then length is 
                               `L=M-1`, otherwise `L=M`.
       hist    <scalar>[L]   : optional, histogram of ``func`` operation on ``weights``
                               returned if ``weights`` is given. 
       std     <scalar>[L]   : optional, standard-deviation of ``weights`` in bin, 
                               returned if ``stdev == True``


    To-Do
    -----
     - Allow multiple ``funcs`` to be performed simultaneously, i.e. 'min' and 'max'
     - Changes ``edges`` options to be 'inner', 'outer', 'left', 'right'

    """

    assert func in [ 'sum', 'ave', 'min', 'max' ], "Invalid ``func`` argument!"

    # For anything besides counting ('sum'), we need a weight for each argument
    if( func is not 'sum' or stdev == True ): 
        assert np.shape(weights) == np.shape(args), "Shape of ``weights`` must match ``args``!"


    ## Prepare Effective bin edges as needed
    #  -------------------------------------

    rightInclusive = False
    useBins = np.array(bins)
    # Design a right-most bin to include right-most particles
    if(   edges == 'left'  ): 
        # Try to go just after right-most value
        useMax  = 1.01*np.max([np.max(useBins), np.max(args)])
        # Deal with right-most being 0.0
        shiftMax = minmax(np.fabs(args), nonzero=True)
        #     If there are no, nonzero values ``None`` is returned
        if( shiftMax is None ): shiftMax = 1.0
        else:                   shiftMax = 0.1*shiftMax[0]
        useMax += shiftMax
        useBins = np.concatenate([useBins, [useMax]])
    # Design a left-most  bin to include left-most  particles
    elif( edges == 'right' ): 
        # Try to go just before left-most value
        useMin  = 0.99*np.min([np.min(useBins), np.min(args)])
        # Deal with left-most being 0.0
        shiftMin = minmax(np.fabs(args), nonzero=True)
        #     If there are no, nonzero values ``None`` is returned
        if( shiftMin is None ): shiftMin = 1.0
        else:                   shiftMin = 0.1*shiftMin[0]
        useMin -= shiftMin
        useBins = np.concatenate([[useMin], useBins])
        rightInclusive = True
    elif( edges == 'both'  ): 
        pass
    else: 
        raise RuntimeError("Unrecognized ``edges`` parameter!!")


    ## Find bins for each value
    #  ------------------------

    # Find where each value belongs
    digits = np.digitize(args, useBins, right=rightInclusive)
    # Histogram values (i.e. count entries in bins)
    counts = [ np.count_nonzero(digits == ii) for ii in range(1, len(useBins)) ]
    # Add values equaling the right-most edge
    if( edges == 'both' ): counts[-1] += np.count_nonzero( args == useBins[-1] )
    
    counts = np.array(counts)

    # Just histogramming counts
    if( weights is None ):
        return counts


    ## Perform Weighting
    #  -----------------

    weights = np.array(weights)

    # if a single scaling is provided
    if( np.size(weights) == 1 ):
        return counts, counts*weights


    # If ``weights`` has values for each argument ``args``

    useFunc = np.sum
    if(   func == 'min' ): useFunc = np.min
    elif( func == 'max' ): useFunc = np.max

    # Sum values in bins
    hist = [ useFunc(weights[digits == ii]) for ii in range(1, len(useBins)) ]
    # Add values equaling the right-most edge
    if( edges == 'both' ): 
        # Add values directly into right-most bin
        if( func == 'ave' or func == 'sum' ): 
            hist[-1] += useFunc( weights[args == useBins[-1]] )
        # Find min/max of values compared to whats already in right-most bin
        else:
            hist[-1]  = useFunc( [hist[-1],weights[args == useBins[-1]]] )

    # Average Bins
    if( func == 'ave' ): hist = [ vv/hh if hh > 0 else 0.0 for hh,vv in zip(counts, hist) ]

    hist = np.array(hist)


    # Calculate standard-deviation of values in each bin
    if( stdev ):

        # Sum values in bins
        std = [ np.std(weights[digits == ii]) for ii in range(1, len(useBins)) ]
        # Fix last bin to include values which equal the right-most edge
        if( edges == 'both' ): 
            std[-1] = np.std( weights[ (digits == ii) | (args == useBins[-1]) ] ) 

        std = np.array(std)

        return counts, hist, std


    # No ``std``, just return histograms of counts and ``func`` on ``weights``
    return counts, hist

# histogram()



def mid(vals, log=False):
    
    mids = np.zeros(len(vals)-1)
    for ii,vv in enumerate(zip(vals[:-1], vals[1:])):
        if( log ): mids[ii] = np.power(10.0, np.average(np.log10(vv)) )
        else:      mids[ii] = np.average(vv)

    return mids

# mid()


def dist(r1, r2):
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

    if( len(np.shape(r1)) > 1 or len(np.shape(r2)) > 1 ):
        dist = np.sqrt( np.sum( np.square(r1 - r2), axis=1) )    
    else:
        dist = np.sqrt( np.sum( np.square(r1 - r2) ) )

    return dist

# dist()


def extend(arr, log=True):
    """
    """

    if( log ): useArr = np.log10(arr)
    else:      useArr = np.array(arr)

    left = useArr[ 0] + (useArr[ 0] - useArr[ 1])
    rigt = useArr[-1] + (useArr[-1] - useArr[-2])

    if( log ):
        left = np.power(10.0, left)
        rigt = np.power(10.0, rigt)

    return left, rigt

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
    conf = np.array([ np.percentile(vals, 100.0*cdf) for cdf in cdf_vals ])
    med = np.percentile(vals, 50.0)

    if( len(conf) == 1 ): conf = conf[0]

    return med, conf

# confidenceIntervals()
