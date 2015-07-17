"""
General plotting functions.

Functions
---------
  - subplots             :
  - set_lim              : set limits on an axis
  - addParameterString   : 
  - unifyAxesLimits      : given a list of axes, set all limits to match flobal extrema
  - colorCycle           : create a cycle of the given number of colors

  - twinAxis             : easily create and set a new twin axis (like `twinx()` or `twiny()`)
  - setAxis              : function to set many different axis properties at once


  - plotHistLine         : plot a line as a histogram
  - skipTicks            : skip some tick marks
  - saveFigure
  - plotSegmentedLine    : Plot a line as a series of segements (e.g. with various colors)
  - colormap             : create a colormap from scalars to colors
  - strSciNot            : create a latex string of the given number in scientific notation



  - _setAxis_scale       : 
  - _setAxis_label       :
  - _histLine            : construct a stepped line
  - _clear_frame
  - _make_segments


"""


import numpy as np
import astropy   as ap
import scipy as sp
import matplotlib as mpl
from matplotlib import pyplot as plt
from datetime import datetime

import Math as zmath


VALID_SIDES = [ None, 'left', 'right', 'top', 'bottom' ]


def subplots(figsize=[14,8], nrows=1, ncols=1, logx=True, logy=True, grid=True, 
             invx=False, invy=False, twinx=False, twiny=False, 
             xlim=None, ylim=None, twinxlim=None, twinylim=None,
             left=0.1, right=0.9, top=0.9, bot=0.1, hspace=0.25, wspace=0.2):

    fig, axes = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols)

    if( not np.iterable(axes) ): axes = [axes]

    for ax in axes:
        if( logx ): ax.set_xscale('log')
        if( logy ): ax.set_yscale('log')
        if( grid ): ax.grid()
        if( invx ): ax.invert_xaxis()
        if( invy ): ax.invert_yaxis()
        if( xlim is not None ): ax.set_xlim(xlim)
        if( ylim is not None ): ax.set_ylim(ylim)

    if( twinx ): 
        twxs = []
        for ax in axes: twxs.append(ax.twinx())
        twxs = np.array(twxs).reshape(np.shape(axes))
        for tw in twxs: 
            if( logy ): tw.set_yscale('log')
            if( twinylim is not None ): tw.set_ylim(twinylim)

        if( len(twxs) == 1 ): twxs = twxs[0]


    if( twiny ): 
        twys = []
        for ax in axes: twys.append(ax.twiny())
        twys = np.array(twys).reshape(np.shape(axes))
        for tw in twys: 
            if( logy ): tw.set_xscale('log')
            if( twinxlim is not None ): tw.set_ylim(twinxlim)

        if( len(twys) == 1 ): twys = twys[0]

    if( len(axes) == 1 ): axes = axes[0]

    if(   twinx and twiny ): return fig, axes, twxs, twys
    elif( twinx ):           return fig, axes, twxs
    elif( twiny ):           return fig, axes, twys
    
    plt.subplots_adjust(left=left, right=right, top=top, bottom=bot, wspace=wspace, hspace=hspace)



    return fig, axes

# subplots()


def set_lim(ax, axis='y', lo=None, hi=None, data=None, range=False, at='exactly'):
    """
    Set the limits (range) of the given, target axis.

    When only ``lo`` or only ``hi`` is specified, the default behavior is to only set that axis
    limit and leave the other bound to its existing value.  When ``range`` is set to `True`, then
    the given axis boumds (``lo``/``hi``) are used as multipliers, i.e.

        >>> Plotting.set_lim(ax, lo=0.1, range=True, at='exactly')
        will set the lower bound to be `0.1` times the existing upper bound

    The ``at`` keyword determines whether the given bounds are treated as limits to the bounds,
    or as fixed ('exact') values, i.e.

        >>> Plotting.set_lim(ax, lo=0.1, range=True, at='most')
        will set the lower bound to at-'most' `0.1` times the existing upper bound.  If the lower
        bound is already 0.05 times the upper bound, it will not be changed.


    Arguments
    ---------
       ax    : <matplotlib.axes.Axes>, base axes object to modify
       axis  : <str>{'x','y'}, which axis to set
       lo    : <scalar>, lower  limit bound
       hi    : <scalar>, higher (upper) limit bound
       data  : <scalar>[N], range of data values from which to use max and min
       range : <bool>, set the 'range' of the axis limits (True) or set the bounds explicitly
       at    : <str>{'least', 'exactly', 'most'}, how to treat the given bounds - limits or exactly

    """

    AT_LEAST   = 'least'
    AT_MOST    = 'most'
    AT_EXACTLY = 'exactly'
    AT_VALID   = [ AT_LEAST, AT_EXACTLY, AT_MOST ]
    assert at in AT_VALID, "``at`` must be in {'%s'}!" % (str(AT_VALID))


    if(   axis == 'y' ): 
        get_lim = ax.get_ylim
        set_lim = ax.set_ylim
    elif( axis == 'x' ): 
        get_lim = ax.get_xlim
        set_lim = ax.set_xlim
    else:
        raise RuntimeError("``axis`` must be either 'x' or 'y'!")

    lims = np.array(get_lim())
    
    ## Set Range/Span of Limits
    if( range ):
        if(   lo is not None ): 
            if(   at is AT_EXACTLY ): lims[0] = lims[1]/lo
            elif( at is AT_LEAST   ): lims[0] = np.min([lims[0], lims[0]/lo])
            elif( at is AT_MOST    ): lims[0] = np.max([lims[0], lims[0]/lo])
        elif( hi is not None ): 
            if(   at is AT_EXACTLY ): lims[1] = lims[1]*hi
            elif( at is AT_LEAST   ): lims[1] = np.min([lims[1], lims[1]*hi])
            elif( at is AT_MOST    ): lims[1] = np.max([lims[1], lims[1]*hi])
        else: 
            raise RuntimeError("``lo`` or ``hi`` must be provided!")

    ## Set Limits explicitly
    else:
        if(   lo   is not None ): 
            if(   at is AT_EXACTLY ): lims[0] = lo
            elif( at is AT_LEAST   ): lims[0] = np.min([lims[0], lo])
            elif( at is AT_MOST    ): lims[0] = np.max([lims[0], lo])
        elif( data is not None ): 
            lims[0] = np.min(data)

        if(   hi   is not None ): 
            if(   at is AT_EXACTLY ): lims[1] = hi
            elif( at is AT_LEAST   ): lims[1] = np.max([lims[1], hi])
            elif( at is AT_MOST    ): lims[1] = np.min([lims[1], hi])
        elif( data is not None ): 
            lims[1] = np.max(data)


    set_lim(lims)

    return

# set_lim()


def addParameterString(fig, pstr, x=0.98, y=0.1, halign='right', valign='bottom', fs=16):
    txt = fig.text(x, y, pstr, size=fs, family='monospace', transform=fig.transFigure,
                   horizontalalignment=halign, verticalalignment=valign)
    return txt


def unifyAxesLimits(axes, axis='y'):
    """
    Given a list of axes, set all limits to match global extrema.
    """

    assert axis in ['x','y'], "``axis`` must be either 'x' or 'y' !!"

    if( axis == 'y' ):
        lims = np.array([ax.get_ylim() for ax in axes])
    else:
        lims = np.array([ax.get_xlim() for ax in axes])

    lo = np.min(lims[:,0])
    hi = np.max(lims[:,1])

    for ax in axes:
        if( axis == 'y' ):
            ax.set_ylim([lo,hi])
        else:
            ax.set_xlim([lo,hi])

    return np.array([lo,hi])

# unifyAxesLimits()


'''
def setLineStyleCycle(num):

    LS_DASH_DASH = [8,4]
    LS_DASH_DOT  = [8,4,4,4]
    LS_DOT_DOT   = [4,4]

    LS_DASH = [8,4]
    LS_DASH_L = [12,4]
    LS_DOT  = [4,4]

    LS = [[None,None]]  # solid
    LS.append(LS_DASH_L)
    LS.append(LS_DASH)
    LS.append(LS_DOT)

    LS.append(LS_DASH_L + LS_DOT)
    LS.append(LS_DASH + LS_DOT)

    LS.append(LS_DASH_L + LS_DASH + LS_DOT )
    LS.append(LS_DASH_L + LS_DOT + LS_DASH + LS_DOT )

    LS.append(LS_DASH_L + LS_DOT + LS_DOT)
    LS.append(LS_DASH + LS_DOT + LS_DOT)

    LS.append(LS_DASH_L + LS_DOT + LS_DOT + LS_DOT)
    LS.append(LS_DASH + LS_DOT + LS_DOT + LS_DOT)

    LS.append(LS_DASH_L + LS_DOT + LS_DOT + LS_DASH + LS_DOT + LS_DOT)

    return LS[:num]
'''

def colorCycle(num, ax=None, cmap=plt.cm.spectral, left=0.1, right=0.9):
    cols = [cmap(it) for it in np.linspace(left, right, num)]
    if( ax is not None ): ax.set_color_cycle(cols[::-1])
    return cols


def twinAxis(ax, axis='x', pos=1.0, **kwargs):

    if(   axis == 'x' ):
        tw = ax.twinx()
        setax = 'y'
    elif( axis == 'y' ):
        tw = ax.twiny()
        setax = 'x'
    else:
        raise RuntimeError("``axis`` must be either {`x` or `y`}!")

 
    tw = setAxis(tw, axis=setax, pos=pos, **kwargs)

    return tw



def setAxis(ax, axis='x', c='black', fs=12, pos=None, trans='axes', label=None, scale=None, 
            thresh=None, side=None, ts=8, grid=True, lim=None, invert=False, ticks=True):
    """
    Configure a particular axis of the given axes object.

    Arguments
    ---------
       ax     : <matplotlib.axes.Axes>, base axes object to modify
       axis   : <str>, which axis to target {``x`` or ``y``}
       c      : <str>, color for the axis (see ``matplotlib.colors``)
       fs     : <int>, font size for labels
       pos    : <float>, position of axis-label/lines relative to the axes object
       trans  : <str>, transformation type for the axes
       label  : <str>, axes label (``None`` means blank)
       scale  : <str>, axis scale, e.g. 'log', (``None`` means default)
       thresh : <float>, for 'symlog' scaling, the threshold for the linear segment
       side   : <str>, where to place the markings, {``left``, ``right``, ``top``, ``bottom``}
       ts     : <int>, tick-size (for the major ticks only)
       grid   : <bool>, whether grid lines should be enabled
       lim    : <float>[2], limits for the axis range
       invert : <bool>, whether to invert this axis direction (i.e. high to low)
       ticks

    """

    assert axis  in ['x','y'],                          "``axis`` must be `x` or `y`!"
    assert trans in ['axes','figure'],                  "``trans`` must be `axes` or `figure`!"
    assert side  in VALID_SIDES, "``side`` must be in '%s'" % (VALID_SIDES)

    # Set tick colors and font-sizes
    ax.tick_params(axis=axis, which='both', colors=c, labelsize=fs)
    #    Set tick-size only for major ticks
    ax.tick_params(axis=axis, which='major', size=ts)

    # Set Grid Lines
    ax.grid(grid, axis=axis)

    if( axis == 'x' ):
        ax.xaxis.label.set_color(c)
        offt = ax.get_xaxis().get_offset_text()

        if( side is None ):
            if( pos is None ):
                side = 'bottom'
            else:
                if( pos < 0.5 ): side = 'bottom'
                else:            side = 'top'

        if( pos is not None ):
            offt.set_y(pos)
            ax.xaxis.set_label_position(side)
            ax.xaxis.set_ticks_position(side)

        if( lim is not None ): 
            if( np.size(lim) > 2 ): lim = zmath.minmax(lim)
            ax.set_xlim( lim )

        if( invert ): ax.invert_xaxis()
        if( not ticks ):
            for tlab in ax.xaxis.get_ticklabels(): tlab.set_visible(False)

    else:
        ax.yaxis.label.set_color(c)
        offt = ax.get_yaxis().get_offset_text()

        if( side is None ):
            if( pos is None ):
                side = 'left'
            else:
                if( pos < 0.5 ): side = 'left'
                else:            side = 'right'


        if( pos is not None ):
            offt.set_x(pos)

        ax.yaxis.set_label_position(side)
        ax.yaxis.set_ticks_position(side)

            
        if( lim is not None ): ax.set_ylim( lim )

        if( invert ): ax.invert_yaxis()
        if( not ticks ):
            for tlab in ax.yaxis.get_ticklabels(): tlab.set_visible(False)


    # Set Spine colors
    ax.spines[side].set_color(c)
    if( pos is not None ):
        ax.set_frame_on(True)
        ax.spines[side].set_position((trans, pos))
        ax.spines[side].set_visible(True)
        ax.patch.set_visible(False)

    # Set Axis Scaling
    if( scale is not None ): _setAxis_scale(ax, axis, scale, thresh=thresh)

    # Set Axis Label
    _setAxis_label(ax, axis, label, fs=fs, c=c)

    offt.set_color(c)

    return ax

# setAxis()



def plotHistLine(ax, edges, hist, yerr=None, nonzero=False, positive=False, extend=None, 
                 fill=None, **kwargs):
    """
    Given bin edges and histogram-like values, plot a histogram.

    Arguments
    ---------
        ax       <obj>    : matplotlib axes object on which to plot
        edges    <flt>[N] : positions of bin edges, length either that of hist ``M`` or ``M+1``
        hist     <flt>[M] : histogram values for each bin
        yerr     <flt>[M] : value for y-error-bars
        nonzero  <bool>   : only plot non-zero values
        positive <bool>   : only plot positive values
        extend   <str>    : required if ``N != M+1``, sets direction to extend ``edges``
        fill     <obj>    : If not ``None``, fill below line; can set as dict of fill parameters
        **kwargs <dict>   : key value pairs to be passed to the plotting function

    Returns
    -------
        line     <obj>    : the plotted object

    """

    # Determine color if included in ``kwargs``
    col = 'black'
    if(   kwargs.get('color') is not None ): col = kwargs.get('color')
    elif( kwargs.get('c')     is not None ): col = kwargs.get('c')
    
    # Extend bin edges if needed
    if( len(edges) != len(hist)+1 ):
        if(   extend == 'left'  ): edges = np.concatenate([[zmath.extend(edges)[0]], edges])
        elif( extend == 'right' ): edges = np.concatenate([edges, [zmath.extend(edges)[1]]])
        else: raise RuntimeError("``edges`` must be longer than ``hist``, or ``extend`` given")
    
    # Construct plot points to manually create a step-plot
    xval, yval = _histLine(edges, hist)

    # Select nonzero values
    if( nonzero ):
        xval = np.ma.masked_where(yval == 0.0, xval)
        yval = np.ma.masked_where(yval == 0.0, yval)

    # Select positive values
    if( positive ):
        xval = np.ma.masked_where(yval < 0.0, xval)
        yval = np.ma.masked_where(yval < 0.0, yval)

    # Plot Histogram
    line, = ax.plot(xval, yval, **kwargs)

    # Add yerror-bars
    if( yerr is not None ): 
        xmid = zmath.mid(edges)

        if( nonzero ): 
            inds = np.where( hist != 0.0 )
            ax.errorbar(xmid[inds], hist[inds], yerr=yerr[inds], fmt=None, ecolor=col)
        else:
            ax.errorbar(xmid,       hist,       yerr=yerr,       fmt=None, ecolor=col)

    # Add a fill region
    if( fill is not None ):
        ylim = ax.get_ylim()
        if( type(fill) == dict ): filldict = fill
        else:                     filldict = dict()

        ax.fill_between(xval, yval, 0.1*ylim[0], **filldict)
        ax.set_ylim(ylim)


    return line

# plotHistLine()    



def skipTicks(ax, axis='y', skip=2, num=None, first=None, last=None):
    """
    Only label every ``skip`` tick marks.

    Arguments
    ---------
        ax    <obj>  : `matplotlib.axes.Axes` object, base axes class
        axis  <str>  : which axis to modify
        skip  <int>  : interval which to skip
        num   <int>  : target number of tick labels (``None`` : used a fixed ``skip``)
        first <bool> : If `True` always show first tick, if `False` never show, otherwise use skip
        last  <bool> : If `True` always show last  tick, if `False` never show, otherwise use skip

    """

    # Get the correct labels
    if(   axis == 'y' ): ax_labels = ax.yaxis.get_ticklabels()
    elif( axis == 'x' ): ax_labels = ax.yaxis.get_ticklabels()
    else: raise RuntimeError("Unrecognized ``axis`` = '%s'!!" % (axis))

    count = len(ax_labels)

    # Determine ``skip`` to match target number of labels
    if( num is not None ): skip = np.int(np.ceil(1.0*count/num))

    visible = np.zeros(count, dtype=bool)

    # Choose some to be visible
    visible[::skip] = True

    if(   first is True  ): visible[ 0] = True
    elif( first is False ): visible[ 0] = False

    if(   last  is True  ): visible[-1] = True
    elif( last  is False ): visible[-1] = False


    for label,vis in zip(ax_labels, visible): label.set_visible(vis)

    return

# skipTicks()



def saveFigure(fname, fig, multipage=False, verbose=True, **kwargs):
    if( multipage ):
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(fname) as pdf:
            for ff in fig: pdf.savefig(figure=ff, **kwargs)

    else:
        fig.savefig(fname, **kwargs)


    if( verbose ): print "Saved figure to '%s'" % (fname)
    return

# saveFigure()




def plotSegmentedLine(ax, xx, yy, zz=None, cmap=plt.cm.jet, norm=[0.0,1.0], lw=3.0, alpha=1.0):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb

    Plot a colored line with coordinates xx and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Get the minimum and maximum of ``norm``
    norm = zmath.minmax(norm)
    # conver to normalization
    norm = plt.Normalize(norm[0], norm[1])
    
    if( zz is None ): zz = np.linspace(norm.vmin, norm.vmax, num=len(xx))
    else:             zz = np.asarray(zz)

    segments = make_segments(xx, yy)
    lc = mpl.collections.LineCollection(segments, array=zz, cmap=cmap, norm=norm, 
                                        linewidth=lw, alpha=alpha)
    
    ax.add_collection(lc)
    
    return lc

# plotSegmentedLine()


def colormap(args, cmap=plt.cm.jet, scale=None):
    """
    Create a colormap from a scalar range to a set of colors.

    Arguments
    ---------
       args  <scalar>([N]) : range of valid scalar values to normalize with
       cmap  <object>      : optional, desired colormap
       scale <str>         : optional, scaling of colormap {'lin', 'log'}

    Returns
    -------
       smap <matplotlib.cm.ScalarMappable> : scalar mappable object which contains the members
                                             ``norm``, ``cmap``, and the function ``to_rgba``

    """

    

    if( scale is None ): 
        if( np.size(args) > 1 ): scale = 'log'
        else:                    scale = 'lin'

    if(   scale.startswith('log') ): log = True
    elif( scale.startswith('lin') ): log = False
    else:
        raise RuntimeError("Unrecognized ``scale`` = '%s'!!" % (scale))


    # Determine minimum and maximum
    if( np.size(args) > 1 ): min,max = zmath.minmax(args, nonzero=log, positive=log)
    else:                    min,max = 0, np.int(args)-1

    # Create normalization
    if( log ): norm = mpl.colors.LogNorm  (vmin=min, vmax=max)
    else:      norm = mpl.colors.Normalize(vmin=min, vmax=max)

    # Create scalar-mappable
    smap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    # Bug-Fix something something
    smap._A = []

    return smap

# colormap()


def strSciNot(val, precman=1, precexp=1):
    """
    """
    man, exp = zmath.frexp10(val)
    str = "${0:.{2:d}f} \\times \, 10^{{ {1:.{3:d}f} }}$" #.format(man, exp, precman, precexp)
    str = str.format(man, exp, precman, precexp)
    return str

# strSciNot()










def _setAxis_scale(ax, axis, scale, thresh=None):

    if( scale.startswith('lin') ): scale = 'linear'

    if( scale == 'symlog' ): thresh = 1.0

    if(   axis == 'x' ): ax.set_xscale(scale, linthreshx=thresh)
    elif( axis == 'y' ): ax.set_yscale(scale, linthreshy=thresh)
    else: raise RuntimeError("Unrecognized ``axis`` = %s" % (axis))
    return


def _setAxis_label(ax, axis, label, fs=12, c='black'):
    if(   axis == 'x' ): ax.set_xlabel(label, size=fs, color=c)
    elif( axis == 'y' ): ax.set_ylabel(label, size=fs, color=c)
    else: raise RuntimeError("Unrecognized ``axis`` = %s" % (axis))
    return


def _histLine(edges, hist):
    xval = np.hstack([ [edges[jj],edges[jj+1]] for jj in range(len(edges)-1) ])
    yval = np.hstack([ [hh,hh] for hh in hist ])
    return xval, yval



def _clear_frame(ax=None): 
    # Taken from a post by Tony S Yu
    if ax is None: ax = plt.gca() 
    ax.xaxis.set_visible(False) 
    ax.yaxis.set_visible(False) 
    for spine in ax.spines.itervalues(): spine.set_visible(False) 
    return

# clear_frame()


def _make_segments(x, y):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb

    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   [numlines, (points per line), 2 (x and y)] array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    return segments

# _make_segments()
