"""
Functions for Math operations.

Functions
---------
 -



"""


import numpy as np
import scipy as sp
import scipy.interpolate


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
