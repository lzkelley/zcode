"""

Functions
---------
-    plot2DHist              - Plot a 2D histogram
-    plot2DHistProj          - Plot a 2D histogram, with projected 1D histograms

"""
import numpy as _np
import matplotlib.pyplot as _plt

import zcode.math as _zmath
from . import core as _zplot

__all__ = ['plot2DHist', 'plot2DHistProj']

_LEFT = 0.08
_RIGHT = 0.06
_BOTTOM = 0.08
_TOP = 0.12

_CB_WID = 0.03
_CB_WPAD = 0.08

_COL = '0.5'


def _constructFigure(fig, figsize, xproj, yproj, hratio, wratio, scale, histScale, labels, cbar):
    """

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure with axes added on (access via ``fig.axes``)
    xkey : int
        The index number of the x-proj plot in the ``fig.axes`` array.  `0` for None.
        e.g. if ``xproj == yproj == True``, then ``xkey = 1`` and ``ykey = 2``.
    ykey : int
        The index number of the y-proj plot in the ``fig.axes`` array.  `0` for None.

    """
    assert 0.0 <= hratio <= 1.0, "`hratio` must be between [0.0,1.0]!"
    assert 0.0 <= wratio <= 1.0, "`wratio` must be between [0.0,1.0]!"

    # Create figure if needed
    if(not fig): fig = _plt.figure(figsize=figsize)

    xpax = ypax = cbax = None

    # Determine usable space and axes sizes
    useable = [1.0-_LEFT-_RIGHT, 1.0-_TOP-_BOTTOM]
    if(cbar):
        useable[0] -= _CB_WID + _CB_WPAD

    if(yproj):
        prim_wid = useable[0]*wratio
        ypro_wid = useable[0]*(1.0-wratio)
    else:
        prim_wid = useable[0]

    if(xproj):
        prim_hit = useable[1]*hratio
        xpro_hit = useable[1]*(1.0-hratio)
    else:
        prim_hit = useable[1]

    # Create primary axes, at bottom left
    prax = fig.add_axes([_LEFT, _BOTTOM, prim_wid, prim_hit])
    prax.set(xscale=scale[0], yscale=scale[1], xlabel=labels[0], ylabel=labels[1])
    _zplot.setGrid(prax, False)

    # Add x-projection axes on top-left
    if(xproj):
        xpax = fig.add_axes([_LEFT, _BOTTOM+prim_hit, prim_wid, xpro_hit])
        xpax.set(xscale=scale[0], yscale=histScale, ylabel='Counts', xlabel=labels[0])
        xpax.xaxis.set_label_position('top')
        _zplot.setGrid(xpax, True, axis='y')

    # Add y-projection axes on bottom-right
    if(yproj):
        ypax = fig.add_axes([_LEFT+prim_wid, _BOTTOM, ypro_wid, prim_hit])
        ypax.set(yscale=scale[1], xscale=histScale, xlabel='Counts', ylabel=labels[1])
        ypax.yaxis.set_label_position('right')
        _zplot.setGrid(ypax, True, axis='x')

    # Add color-bar axes on the right
    if(cbar):
        cbar_left = _LEFT + prim_wid + _CB_WPAD
        if(yproj): cbar_left += ypro_wid
        cbax = fig.add_axes([cbar_left, _BOTTOM, _CB_WID, prim_hit])

    return fig, prax, xpax, ypax, cbax
# } def _constructFigure


def plot2DHistProj(xvals, yvals, bins=10, fig=None, figsize=[18, 12], xproj=True, yproj=True,
                   hratio=0.7, wratio=0.7, fs=12, scale='log', histScale='log', labels=None,
                   cbar=True):
    """
    Plot a 2D histogram with projections of one or both axes.

    Arguments
    ---------
    xvals : array_like, shape (N,)
        Values corresponding to the x-points of the given data
    yvals : array_like, shape (N,)
        Values corresponding to the y-points of the given data
    bins : int or [int, int] or array_like or [array, array], optional
        Specification for bin sizes.  integer values are treated as the number of bins to use,
        while arrays are used as the bin edges themselves.  If a tuple of two values is given, it
        is assumed that the first is for the x-axis and the second for the y-axis.
    fig : matplotlib.figure.figure, optional
        Figure instance to which axes are added for plotting.  One is created if not given.
    figsize : [scalar, scalar], optional
        Size of figure to create, if one is not given.
    xproj : bool, optional
        Whether to also plot the projection of the x-axis (i.e. histogram ignoring y-values).
    yproj : bool, optional
        Whether to also plot the projection of the y-axis (i.e. histogram ignoring x-values).
    hratio : float, optional
        Fraction of the total available height-space to use for the primary axes object (2D hist)
    wratio : float, optional
        Fraction of the total available width-space to use for the primary axes object (2D hist)
    fs : int, optional
        Font-size
    scale : str or [str,str], optional
        Specification for the axes scaling {'log','lin'}.  If two values are given, the first is
        used for the x-axis and the second for the y-axis.
    histScale : str, optional
        Scaling to use for the histograms {'log','lin'}-- the color scale on the 2D histogram,
        or the Counts axis on the 1D histograms.
    cbar : bool, optional
        Add a colorbar.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing plots.

    """
    if(_np.size(scale) == 1):
        scale = [scale, scale]
    elif(_np.size(scale) != 2):
        raise ValueError("`scale` must be one or two scaling specifications!")

    if(not labels): labels = ['', '']

    # Create and initializae figure and axes
    fig, prax, xpax, ypax, cbax = _constructFigure(fig, figsize, xproj, yproj, hratio, wratio,
                                                   scale, histScale, labels, cbar)

    # Create bins
    # -----------
    #     `bins` is a single scalar value -- apply to both
    if(_np.ndim(bins) == 0):
        xbins = bins
        ybins = bins
    else:
        #     `bins` is a pair of bin specifications, separate and apply
        if(len(bins) == 2):
            xbins = bins[0]
            ybins = bins[1]
        #     `bins` is a single array -- apply to both
        elif(len(bins) > 2):
            xbins = bins
            ybins = bins
        #     unrecognized option -- error
        else:
            raise ValueError("Unrecognized shape of ``bins`` = %s" % (str(_np.shape(bins))))

    # If a number of bins is given, create an appropriate spacing
    if(_np.ndim(xbins) == 0):
        xbins = _zmath.spacing(xvals, num=xbins+1, scale=scale[0])

    if(_np.ndim(ybins) == 0):
        ybins = _zmath.spacing(yvals, num=ybins+1, scale=scale[1])

    # Plot 2D Histogram and Projections
    # ---------------------------------
    hist, xedges, yedges = _np.histogram2d(xvals, yvals, bins=[xbins, ybins])
    plot2DHist(prax, xedges, yedges, hist, cbax=cbax)
    # cb = _plt.colorbar(hist_im, orientation='horizontal', cax=axCB)

    # Plot projection of the x-axis (i.e. ignore 'y')
    if(xpax):
        islog = scale[0].startswith('log')
        #     create and plot histogram
        nums, edges, patches = xpax.hist(xvals, bins=xbins, color=_COL, log=islog)
        #     set tick-labels to the top
        _plt.setp(xpax.get_yticklabels(), fontsize=fs)
        xpax.xaxis.tick_top()
        #     set bounds to bin edges
        _zplot.set_lim(xpax, 'x', data=xedges)

    # Plot projection of the y-axis (i.e. ignore 'x')
    if(ypax):
        islog = scale[1].startswith('log')
        #    create and plot histogram
        nums, edges, patches = ypax.hist(
            yvals, bins=ybins, orientation='horizontal', color=_COL, log=islog)
        #     set tick-labels to the top
        _plt.setp(ypax.get_yticklabels(), fontsize=fs, rotation=90)
        ypax.yaxis.tick_right()
        #     set bounds to bin edges
        _zplot.set_lim(ypax, 'y', data=yedges)

    return fig
# } def plot2DHistProj


def plot2DHist(ax, xvals, yvals, hist, cbax=None, cscale='log', cmap=_plt.cm.jet, fs=12,
               extrema=None, **kwargs):
    """
    Plot the given 2D histogram of data.

    Use with (e.g.) ``numpy.histogram2d``,

    Arguments
    ---------
        ax     <obj>      : <matplotlib.axes.Axes> object on which to plot.
        xvals  <flt>[N]   : positions (edges) of x values, assumed to be the same for all rows of ``data``

        cbax   <obj>      : <matplotlib.axes.Axes> object on which to add a colorbar.
        cscale <str>      : scale to use for the colormap {'linear','log'}
        cmap   <obj>      : <matplotlib.colors.Colormap> object to use as colormap

    """

    xgrid, ygrid = _np.meshgrid(xvals, yvals)

    if(extrema is None): extrema = _zmath.minmax(hist, nonzero=(cscale == 'log'))

    # Plot
    smap = _zplot.colormap(extrema, cmap=cmap, scale=cscale)
    pcm = ax.pcolormesh(xgrid, ygrid, hist.T, norm=smap.norm, cmap=smap.cmap, **kwargs)

    # Add color bar
    if(cbax is not None):
        cbar = _plt.colorbar(smap, cax=cbax)
        cbar.set_label('Counts', fontsize=fs)
        cbar.ax.tick_params(labelsize=fs)

    _zplot.set_lim(ax, 'x', data=xvals)
    _zplot.set_lim(ax, 'y', data=yvals)

    return pcm
# } def plot2DHist
