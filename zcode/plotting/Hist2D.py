"""

Functions
---------
-    plot2DHist              - Plot a 2D histogram
-    plot2DHistProj          - Plot a 2D histogram, with projected 1D histograms

"""
import numpy as _np
import matplotlib.pyplot as _plt

import zcode.Math as _zmath
import zcode.Plotting as _zplot

__all__ = ['plot2DHist', 'plot2DHistProj']

_LEFT = 0.1
_RIGHT = 0.05
_BOTTOM = 0.1
_TOP = 0.1

_COL = '0.5'


def _constructFigure(fig, figsize, xproj, yproj, hratio, wratio, scale, histScale):
    assert 0.0 <= hratio <= 1.0, "`hratio` must be between [0.0,1.0]!"
    assert 0.0 <= wratio <= 1.0, "`wratio` must be between [0.0,1.0]!"

    # Create figure if needed
    if(not fig): fig = _plt.figure(figsize=figsize)

    xkey = ykey = 0

    # Determine usable space and axes sizes
    useable = [1.0-_LEFT-_RIGHT, 1.0-_TOP-_BOTTOM]
    if(yproj):
        prim_wid = useable[0]*wratio
        ypro_wid = useable[0]*(1.0-wratio)

    if(xproj):
        prim_hit = useable[1]*hratio
        xpro_hit = useable[1]*(1.0-hratio)

    # Create primary axes, at bottom left
    fig.add_axes([_LEFT, _BOTTOM, prim_wid, prim_hit])
    fig.axes[0].set_xscale(scale[0])
    fig.axes[0].set_yscale(scale[1])
    count = 1
    # Add x-projection axes on top-left
    if(xproj):
        fig.add_axes([_LEFT, _BOTTOM+prim_hit, prim_wid, xpro_hit])
        fig.axes[count].set_xscale(scale[0])
        fig.axes[count].set_yscale(histScale)
        xkey = count
        count += 1

    # Add y-projection axes on bottom-right
    if(yproj):
        fig.add_axes([_LEFT+prim_wid, _BOTTOM, ypro_wid, prim_hit])
        fig.axes[count].set_yscale(scale[1])
        fig.axes[count].set_xscale(histScale)
        ykey = count

    for ax in fig.axes:
        _zplot.setGrid(ax, False)

    return fig, xkey, ykey


def plot2DHistProj(xvals, yvals, bins=10, fig=None, figsize=[16, 12], xproj=True, yproj=True,
                   hratio=0.7, wratio=0.7, fs=12, scale='log', histScale='log'):
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

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing plots.

    """
    if(_np.size(scale) == 1):
        scale = [scale, scale]
    elif(_np.size(scale) != 2):
        raise ValueError("`scale` must be one or two scaling specifications!")

    # Create and initializae figure and axes
    fig, xk, yk = _constructFigure(fig, figsize, xproj, yproj, hratio, wratio, scale, histScale)
    axes = fig.axes

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
    plot2DHist(axes[0], xedges, yedges, hist)
    # cb = _plt.colorbar(hist_im, orientation='horizontal', cax=axCB)

    # Plot projection of the x-axis (i.e. ignore 'y')
    if(xproj):
        islog = scale[0].startswith('log')
        #     create and plot histogram
        nums, edges, patches = axes[xk].hist(xvals, bins=xbins, color=_COL, log=islog)
        #     set tick-labels to the top
        _plt.setp(axes[xk].get_yticklabels(), fontsize=fs)
        axes[xk].xaxis.tick_top()
        #     set bounds to bin edges
        _zplot.set_lim(axes[xk], 'x', data=xedges)

    # Plot projection of the y-axis (i.e. ignore 'x')
    if(yproj):
        islog = scale[1].startswith('log')
        #    create and plot histogram
        nums, edges, patches = axes[yk].hist(
            yvals, bins=ybins, orientation='horizontal', color=_COL, log=islog)
        #     set tick-labels to the top
        _plt.setp(axes[yk].get_yticklabels(), fontsize=fs, rotation=90)
        axes[yk].yaxis.tick_right()
        #     set bounds to bin edges
        _zplot.set_lim(axes[yk], 'y', data=yedges)

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
