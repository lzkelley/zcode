"""Methods for plotting 2D histograms with optional 1D projected histograms for each axis.

Functions
---------
-    plot2DHistProj          - Plot a 2D histogram with projections of one or both axes.
-    plot2DHist              - Plot the given 2D histogram of data.

-    _constructFigure        - Add the required axes to the given figure object.

To-do
-----
-   `plot2DHistProj` and `plot2DHist` use different styles of input data, specifically
    `plot2DHist` requires a 2D histogram of data to be passed, while `plot2DHistProj` accepts
    separate arrays of x and y data to be histogrammed using ``scipy.stats.binned_statistic_2d``.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import scipy as sp
import scipy.stats
import matplotlib.pyplot as plt

import zcode.math as zmath
from . import plot_core

__all__ = ['plot2DHist', 'plot2DHistProj']

_LEFT = 0.08
_RIGHT = 0.06
_BOTTOM = 0.08
_TOP = 0.12

_CB_WID = 0.03
_CB_WPAD = 0.08

_COL = '0.5'


def plot2DHistProj(xvals, yvals, weights=None, statistic=None, bins=10,
                   fig=None, xproj=True, yproj=True, hratio=0.7, wratio=0.7, pad=0.0,
                   fs=12, scale='log', histScale='log', labels=None, cbar=True):
    """Plot a 2D histogram with projections of one or both axes.

    Arguments
    ---------
    xvals : (N,) array_like,
        Values corresponding to the x-points of the given data
    yvals : (N,) array_like,
        Values corresponding to the y-points of the given data
    weights : (N,) array_like or `None`,
        Weights used to create histograms.  If `None`, then counts are used.
    statistic : str or `None`,
        Type of statistic to be calculated, passed to ``scipy.stats.binned_statistic``.
        e.g. {'count', 'sum', 'mean'}.
        If `None`, then either 'sum' or 'count' is used depending on if `weights` are
        provieded or not.
    bins : int or [int, int] or array_like or [array, array],
        Specification for bin sizes.  integer values are treated as the number of bins to use,
        while arrays are used as the bin edges themselves.  If a tuple of two values is given, it
        is assumed that the first is for the x-axis and the second for the y-axis.
    fig : ``matplotlib.figure.figure``,
        Figure instance to which axes are added for plotting.  One is created if not given.
    xproj : bool,
        Whether to also plot the projection of the x-axis (i.e. histogram ignoring y-values).
    yproj : bool,
        Whether to also plot the projection of the y-axis (i.e. histogram ignoring x-values).
    hratio : float,
        Fraction of the total available height-space to use for the primary axes object (2D hist)
    wratio : float,
        Fraction of the total available width-space to use for the primary axes object (2D hist)
    pad : float,
        Padding between central axis and the projected ones.
    fs : int,
        Font-size
    scale : str or [str, str],
        Specification for the axes scaling {'log','lin'}.  If two values are given, the first is
        used for the x-axis and the second for the y-axis.
    histScale : str,
        Scaling to use for the histograms {'log','lin'}-- the color scale on the 2D histogram,
        or the Counts axis on the 1D histograms.
    cbar : bool,
        Add a colorbar.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing plots.

    """
    if np.size(scale) == 1:
        scale = [scale, scale]
    elif np.size(scale) != 2:
        raise ValueError("`scale` must be one or two scaling specifications!")

    if not labels: labels = ['', '']
    scale = [plot_core._clean_scale(sc) for sc in scale]

    if statistic is None:
        if weights is None: statistic = 'count'
        else:               statistic = 'sum'

    # Create and initializae figure and axes
    fig, prax, xpax, ypax, cbax = _constructFigure(fig, xproj, yproj, hratio, wratio, pad,
                                                   scale, histScale, labels, cbar)

    # Create bins
    # -----------
    #     `bins` is a single scalar value -- apply to both
    if np.isscalar(bins):
        xbins = bins
        ybins = bins
    else:
        #     `bins` is a pair of bin specifications, separate and apply
        if len(bins) == 2:
            xbins = bins[0]
            ybins = bins[1]
        #     `bins` is a single array -- apply to both
        elif len(bins) > 2:
            xbins = bins
            ybins = bins
        #     unrecognized option -- error
        else:
            raise ValueError("Unrecognized shape of ``bins`` = %s" % (str(np.shape(bins))))

    # If a number of bins is given, create an appropriate spacing
    if np.ndim(xbins) == 0:
        xbins = zmath.spacing(xvals, num=xbins+1, scale=scale[0])

    if np.ndim(ybins) == 0:
        ybins = zmath.spacing(yvals, num=ybins+1, scale=scale[1])

    # Plot 2D Histogram and Projections
    # ---------------------------------
    hist, xedges, yedges, binnums = sp.stats.binned_statistic_2d(
        xvals, yvals, weights, statistic=statistic, bins=[xbins, ybins])
    hist = np.nan_to_num(hist)
    pcm, smap = plot2DHist(prax, xedges, yedges, hist, cbax=cbax, labels=labels)

    # Plot projection of the x-axis (i.e. ignore 'y')
    if xpax:
        islog = scale[0].startswith('log')
        #     create and plot histogram
        hist, edges, bins = sp.stats.binned_statistic(
            xvals, weights, statistic=statistic, bins=xbins)
        xpax.bar(edges[:-1], hist, color=smap.to_rgba(hist), log=islog, width=np.diff(edges))
        #     set tick-labels to the top
        plt.setp(xpax.get_yticklabels(), fontsize=fs)
        xpax.xaxis.tick_top()
        #     set bounds to bin edges
        plot_core.setLim(xpax, 'x', data=xedges)

    # Plot projection of the y-axis (i.e. ignore 'x')
    if ypax:
        islog = scale[1].startswith('log')
        #    create and plot histogram
        hist, edges, bins = sp.stats.binned_statistic(
            yvals, weights, statistic=statistic, bins=ybins)
        ypax.barh(edges[:-1], hist, color=smap.to_rgba(hist), log=islog, height=np.diff(edges))
        #     set tick-labels to the top
        plt.setp(ypax.get_yticklabels(), fontsize=fs, rotation=90)
        ypax.yaxis.tick_right()
        #     set bounds to bin edges
        plot_core.setLim(ypax, 'y', data=yedges)

    return fig


def plot2DHist(ax, xvals, yvals, hist, cbax=None, cscale='log', cmap=plt.cm.jet, fs=12,
               extrema=None, labels=None, **kwargs):
    """Plot the given 2D histogram of data.

    Use with (e.g.) ``numpy.histogram2d``,

    Arguments
    ---------
        ax : ``matplotlib.axes.Axes`` object
            Axes object on which to plot.
        xvals : (N,) array of scalars
            Positions (edges) of x values, assumed to be the same for all rows of
            the input data `hist`.
        yvals : (M,) array of scalars
            Positions (edges) of y values, assumed to be the same for all columns of
            the input data `hist`.
        cbax : ``matplotlib.axes.Axes`` object
            Axes object on which to add a colorbar.
        cscale : str
            Scale to use for the colormap {'linear', 'log'}.
        cmap : ``matplotlib.colors.Colormap`` object
            Matplotlib colormap to use for coloring histogram.

    Returns
    -------
        pcm : ``matplotlib.collections.QuadMesh`` object
            The resulting plotted object, returned by ``ax.pcolormesh``.
        smap : ``matplotlib.cm.ScalarMappable`` object
            Colormap and color-scaling information.  See: ``zcode.plot.plot_core.colormap``.

    """

    xgrid, ygrid = np.meshgrid(xvals, yvals)

    if(extrema is None): extrema = zmath.minmax(hist, nonzero=(cscale.startswith('log')))
    if(labels is not None and np.size(labels) > 2):
        cblab = labels[2]
    else:
        cblab = 'Counts'

    # Plot
    smap = plot_core.colormap(extrema, cmap=cmap, scale=cscale)
    pcm = ax.pcolormesh(xgrid, ygrid, hist.T, norm=smap.norm, cmap=smap.cmap, **kwargs)

    # Add color bar
    if(cbax is not None):
        cbar = plt.colorbar(smap, cax=cbax)
        cbar.set_label(cblab, fontsize=fs)
        cbar.ax.tick_params(labelsize=fs)

    plot_core.setLim(ax, 'x', data=xvals)
    plot_core.setLim(ax, 'y', data=yvals)

    return pcm, smap


def _constructFigure(fig, xproj, yproj, hratio, wratio, pad, scale, histScale, labels, cbar):
    """Add the required axes to the given figure object.

    Arguments
    ---------
    ...

    Returns
    -------
    fig : ``matplotlib.figure.Figure`` object,
        Figure with added axes.
    prax : ``matplotlib.axes.Axes`` object,
        Primary 2D histogram axes.  Always created.
    xpax : ``matplotlib.axes.Axes`` object or `None`,
        Projection of the x-axis, if ``xproj == True``.  i.e. y-axis marginalized over.
    ypax : ``matplotlib.axes.Axes`` object or `None`,
        Projection of the y-axis, if ``yproj == True``.  i.e. x-axis marginalized over.
    cbax : ``matplotlib.axes.Axes`` object or `None`,
        Axes for the colorbar, if ``cbar == True``.

    """
    assert 0.0 <= hratio <= 1.0, "`hratio` must be between [0.0, 1.0]!"
    assert 0.0 <= wratio <= 1.0, "`wratio` must be between [0.0, 1.0]!"

    # Create figure if needed
    if(not fig): fig = plt.figure()

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
    #    d
    prax = fig.add_axes([_LEFT, _BOTTOM, prim_wid, prim_hit])
    prax.set(xscale=scale[0], yscale=scale[1], xlabel=labels[0], ylabel=labels[1])
    plot_core.setGrid(prax, False)

    if(len(labels) > 2): histLab = labels[2]
    else:                histLab = 'Counts'

    # Add x-projection axes on top-left
    if(xproj):
        xpax = fig.add_axes([_LEFT, _BOTTOM+prim_hit+pad, prim_wid, xpro_hit-pad])
        xpax.set(xscale=scale[0], yscale=histScale, ylabel=histLab, xlabel=labels[0])
        xpax.xaxis.set_label_position('top')
        plot_core.setGrid(xpax, True, axis='y')

    # Add y-projection axes on bottom-right
    if(yproj):
        ypax = fig.add_axes([_LEFT+prim_wid+pad, _BOTTOM, ypro_wid-pad, prim_hit])
        ypax.set(yscale=scale[1], xscale=histScale, xlabel=histLab, ylabel=labels[1])
        ypax.yaxis.set_label_position('right')
        plot_core.setGrid(ypax, True, axis='x')

    # Add color-bar axes on the right
    if(cbar):
        cbar_left = _LEFT + prim_wid + _CB_WPAD
        if(yproj): cbar_left += ypro_wid
        cbax = fig.add_axes([cbar_left, _BOTTOM, _CB_WID, prim_hit])

    return fig, prax, xpax, ypax, cbax
