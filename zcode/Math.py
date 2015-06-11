"""
Functions for Math operations.

Functions
---------
 -



"""


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



def withinBounds(val, arr, edges=True):
    """
    Test whether a value or array is within the bounds of another.

    Arguments
    ---------
       val   : <scalar>([N]), test value(s)
       arr   : <scalar>[M],   array or span to compare with
       edges : (<bool>), include the boundaries of ``arr`` as 'within'

    Returns
    -------
       <bool>, True if within, False otherwise

    """

    # Including edges (thus equal to edges is 'inside' )
    if( edges ):
        if( np.min(val) <  np.min(arr) ): return False
        if( np.max(val) >  np.max(arr) ): return False
    # Excluding edges (thus equal to edges is 'outside')
    else:
        if( np.min(val) <= np.min(arr) ): return False
        if( np.max(val) >= np.max(arr) ): return False

    return True


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

    """

    # Filter out zeros if desired
    if( nonzero ): useData = np.array(data[np.nonzero(data)])
    else:          useData = np.array(data)

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

    usedata = np.array(data)
    if( nonzero  ): usedata = usedata[np.nonzero(usedata)]
    if( positive ): usedata = usedata[np.where(usedata > 0.0)]

    span = minmax(usedata)
    if(   scale.startswith('log') ): spacing = np.logspace( *np.log10(span), num=num )
    elif( scale.startswith('lin') ): spacing = np.linspace( span,            num=num )
    else: raise RuntimeError("``scale`` unrecognized!")

    return spacing

# spacing()



def histogram(args, bins, weights=None, scale=None, ave=False, edges='both'):
    """
    Histogram (bin) the given values.

    Arguments
    ---------
       args    <scalar>[N]   :
       bins    <scalar>[M]   : 
       weights <scalar>[N]   : optional, weighting factors for each input value in ``args``
       scale   <scalar>([M]) : optional, factor with which to scale each resulting bin
       ave     <bool>        : optional, average over each bin instead of summing
       edges   <str>         : optional, how to treat the given ``bins`` values, i.e. which 'edges'
                               these represent.  Must be one of {`left`, `both`, `right`}.
                               ``left``  : then ``bins[0]`` is the left-most inclusive edge, and
                                           there is no right-most edge
                               ``right`` : then ``bins[-1]`` is the right-most inclusive edge, and
                                           there is no  left-most edge
                               ``both``  : then values outside of the range of ``bins`` will not be
                                           counted anywhere.  Returned histogram will have length
                                           `M-1` --- representing space between ``bins`` values

    Returns
    -------
       hist <scalar>[L] : resulting histogram, if ``edges`` is `both` then length is `L=M-1`,
                          otherwise `L=M`.

    """

    if( ave ): assert weights is not None, "``weights`` must be provided to average!"


    # Prepare Effective bin edges as needed
    rightInclusive = False
    if(   edges == 'left'  ): 
        useBins = np.concatenate([bins, [1.01*np.max(args)]])
    elif( edges == 'right' ): 
        useBins = np.concatenate([[0.99*np.min(args)], bins])
        rightInclusive = True
    elif( edges == 'both'  ): 
        useBins = np.array(bins)
    else: 
        raise RuntimeError("Unrecognized ``edges`` parameter!!")


    # Find where each value belongs
    digits = np.digitize(args, useBins, right=rightInclusive)

    # Histogram values (i.e. count entries in bins)
    hist = [ np.count_nonzero(digits == ii) for ii in range(1, len(useBins)) ]
    # Add values equaling the right-most edge
    if( edges == 'both' ): hist[-1] += np.count_nonzero( args == useBins[-1] )

    if( weights is not None ):
        # Sum values in bins
        vals = [ np.sum(weights[digits == ii]) for ii in range(1, len(useBins)) ]
        # Add values equaling the right-most edge
        if( edges == 'both' ): sums[-1] += np.weights( [args == useBins[-1]] )

        # Average Bins
        if( ave ): hist = [ vv/hh if hh > 0 else 0.0 for hh,vv in zip(hist, vals) ]
        else:      hist = vals


    # Convert to numpy array
    hist = np.array(hist)

    # Rescale the bin values
    if( scale is not None ): hist *= scale

    return hist

# histogram()
