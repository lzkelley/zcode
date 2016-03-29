"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import warnings

from . import math_core

__all__ = ['histogram']


def histogram(args, bins, scale=None, bounds='both', weights=None, func='sum',
              cumul=False, stdev=False):
    """Histogram (bin) the given values.

    Currently `bins` must be monotonically increasing!!
    When using ``bounds=='both'``, you currently can't control which interior bin edges are
    inclusive.

    Arguments
    ---------
       args     <scalar>[N]   : data to be histogrammed.
       bins     <scalar>([M]) : Positions of bin edges or number of bins to construct automatically.
                                If a single ``bins`` value is given, it is assumed to be a number of
                                bins to create with a scaling given by ``scale`` (see desc).
       scale <str>            : scaling to use when creating bins automatically.
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

    assert func in ['sum', 'ave', 'min', 'max'], "Invalid ``func`` argument!"
    warnings.warn("Can you use ``scipy.stats.binned_statistic`` instead?!")

    # For anything besides counting ('sum'), we need a weight for each argument
    if(func != 'sum' or stdev):
        assert np.shape(weights) == np.shape(args), "Shape of ``weights`` must match ``args``!"

    # Prepare Effective bin edges as needed
    # -------------------------------------
    rightInclusive = False

    # Construct a certain number ``bins`` bins spaced appropriately
    if(np.size(bins) == 1):
        extr = math_core.minmax(args)
        useScale = scale
        # If no bin scaling is given, make an appropriate choice
        if(useScale is None):
            useScale = 'log'
            # Dont use log if there are zero/negative values
            if(extr[0] <= 0.0):         useScale = 'lin'
            # Dont use log for small ranges of values
            elif(extr[1]/extr[0] < 10.0): useScale = 'lin'

        edges = math_core.spacing(extr, scale=useScale, num=bins+1, nonzero=False)
    else:
        edges = np.array(bins)

    # Design a right-most bin to include right-most particles
    if(bounds == 'left'):
        # Try to go just after right-most value
        useMax = 1.01*np.max([np.max(edges), np.max(args)])
        # Deal with right-most being 0.0
        shiftMax = math_core.minmax(np.fabs(args), filter='!=')
        #     If there are no, nonzero values ``None`` is returned
        if(shiftMax is None): shiftMax = 1.0
        else:                   shiftMax = 0.1*shiftMax[0]
        useMax += shiftMax
        edges = np.concatenate([edges, [useMax]])
    # Design a left-most  bin to include left-most  particles
    elif(bounds == 'right'):
        # Try to go just before left-most value
        useMin = 0.99*np.min([np.min(edges), np.min(args)])
        # Deal with left-most being 0.0
        shiftMin = math_core.minmax(np.fabs(args), filter='!=')
        #     If there are no, nonzero values ``None`` is returned
        if(shiftMin is None): shiftMin = 1.0
        else:                   shiftMin = 0.1*shiftMin[0]
        useMin -= shiftMin
        edges = np.concatenate([[useMin], edges])
        rightInclusive = True
    elif(bounds == 'both'):
        pass
    else:
        raise RuntimeError("Unrecognized ``bounds`` parameter!!")

    # Find bins for each value
    #  ------------------------

    # Find where each value belongs
    digits = np.digitize(args, edges, right=rightInclusive)
    # Histogram values (i.e. count entries in bins)
    counts = [np.count_nonzero(digits == ii) for ii in range(1, len(edges))]
    # Add values equaling the right-most edge
    if(bounds == 'both'): counts[-1] += np.count_nonzero(args == edges[-1])

    counts = np.array(counts)

    # Calculate cumulative distribution
    if(cumul): cumsum = np.cumsum(counts)

    # Just histogramming counts
    if(weights is None):
        if(cumul): return edges, counts, cumsum
        else:      return edges, counts

    # Perform Weighting
    # -----------------

    weights = np.array(weights)

    # if a single scaling is provided
    if(np.size(weights) == 1):
        if(cumul): return edges, counts, cumsum, counts*weights
        else:      return edges, counts, counts*weights

    # If ``weights`` has values for each argument ``args``

    useFunc = np.sum
    if(func == 'min'): useFunc = np.min
    elif(func == 'max'): useFunc = np.max

    # Sum values in bins
    hist = [useFunc(weights[digits == ii]) for ii in range(1, len(edges))]
    # Add values equaling the right-most edge
    if(bounds == 'both'):
        # Add values directly into right-most bin
        if(func == 'ave' or func == 'sum'):
            hist[-1] += useFunc(weights[args == edges[-1]])
        # Find min/max of values compared to whats already in right-most bin
        else:
            hist[-1] = useFunc([hist[-1], weights[args == edges[-1]]])

    # Average Bins
    if(func == 'ave'): hist = [vv/hh if hh > 0 else 0.0 for hh, vv in zip(counts, hist)]

    hist = np.array(hist)

    # Calculate standard-deviations
    # -----------------------------

    if(stdev):
        # Sum values in bins
        std = [np.std(weights[digits == ii]) for ii in range(1, len(edges))]
        # Fix last bin to include values which equal the right-most edge
        if(bounds == 'both'):
            std[-1] = np.std(weights[(digits == len(edges)-1) | (args == edges[-1])])

        std = np.array(std)

        if(cumul): return edges, counts, cumsum, hist, std
        else:      return edges, counts, hist, std

    # No ``std``, just return histograms of counts and ``func`` on ``weights``
    if(cumul): return edges, counts, cumsum, hist

    return edges, counts, hist
