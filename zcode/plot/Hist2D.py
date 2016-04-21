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
from six.moves import xrange

import numpy as np
import scipy as sp
import scipy.stats
import matplotlib.pyplot as plt

import zcode.math as zmath
from . import plot_core

__all__ = ['plot2DHist', 'plot2DHistProj']

_LEFT = 0.09
_RIGHT = 0.92     # Location of right of plots
_BOTTOM = 0.09
_TOP = 0.90       # Location of top of plots

_CB_WID = 0.02
_CB_WPAD = 0.08


def plot2DHistProj(xvals, yvals, weights=None, statistic=None, bins=10, filter=None, extrema=None,
                   fig=None, xproj=True, yproj=True, hratio=0.7, wratio=0.7, pad=0.0, alpha=1.0,
                   cmap=None, smap=None, type='hist',
                   fs=12, scale='log', histScale='log', labels=None, cbar=True, write_counts=False,
                   left=_LEFT, bottom=_BOTTOM, right=_RIGHT, top=_TOP):
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
    filter : str or `None`, or [2,] tuple of str or `None`, or [3,] tubple of str or `None`
        String specifying how to filter the input `data` relative to zero.
        If this is a single value, it is applies to both `xvals` and `yvals`.
        If this is a tuple/list of two values, they correspond to `xvals` and `yvals` respectively.
        If `weights` are provided, the tuple/list should have three values.
    extrema :
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
    cmap : ``matplotlib.colors.Colormap`` object
        Matplotlib colormap to use for coloring histogram.
        Overridden if `smap` is provided.
    smap : `matplotlib.cm.ScalarMappable` object or `None`
        A scalar-mappable object to use for colormaps, or `None` for one to be created.
    type : str, {'hist', 'scatter'}
        What type of plot should be in the center, a 2D Histogram or a scatter-plot.
    fs : int,
        Font-size
    scale : str or [str, str],
        Specification for the axes scaling {'log','lin'}.  If two values are given, the first is
        used for the x-axis and the second for the y-axis.
    histScale : str,
        Scaling to use for the histograms {'log','lin'}-- the color scale on the 2D histogram,
        or the Counts axis on the 1D histograms.
    labels : (2,) str
    cbar : bool,
        Add a colorbar.
    write_counts : bool
        Print a str on each bin writing the number of values included in that bin.
    left : float {0.0, 1.0}
        Location of the left edge of axes relative to the figure.
    bottom : float {0.0, 1.0}
        Location of the bottom edge of axes relative to the figure.
    right : float {0.0, 1.0}
        Location of the right edge of axes relative to the figure.
    top : float {0.0, 1.0}
        Location of the top edge of axes relative to the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing plots.

    """
    # Make sure shapes of input arrays are valid
    if np.shape(xvals) != np.shape(yvals):
        raise ValueError("Shape of `xvals` ({}) must match `yvals` ({}).".format(
            np.shape(xvals), np.shape(yvals)))
    if weights is not None and np.shape(weights) != np.shape(xvals):
        raise ValueError("Shape of `weights` ({}) must match `xvals` and `yvals` ({}).".format(
            np.shape(weights), np.shape(xvals)))

    # Make sure the given `scale` is valid
    if np.size(scale) == 1:
        scale = [scale, scale]
    elif np.size(scale) != 2:
        raise ValueError("`scale` must be one or two scaling specifications!")

    # Check the `labels`
    if labels is None: labels = ['', '', '']
    elif np.size(labels) == 2: labels = [labels[0], labels[1], '']

    if np.size(labels) != 3:
        raise ValueError("`labels` = '{}' is invalid.".format(labels))

    # Make sure scale strings are matplotlib-compliant
    scale = [plot_core._clean_scale(sc) for sc in scale]

    # Determine type of central plot
    if type.startswith('hist'):
        type_hist = True
    elif type.startswith('scat'):
        type_hist = False
        cblabel = str(labels[2])
        labels[2] = 'Count'
    else:
        raise ValueError("`type` = '{}', must be either 'hist', or 'scatter'.".format(type))

    # Infer default statistic
    if statistic is None:
        if weights is None: statistic = 'count'
        else:               statistic = 'sum'

    if filter is None and histScale.startswith('log'):
        filter = 'g'

    # Filter input data
    if filter is not None:
        # Make sure `filter` is an iterable pair
        if weights is None: num = 2
        else:               num = 3

        if not np.iterable(filter): filter = num*[filter]
        elif len(filter) == 1: filter = num*[filter[0]]

        if len(filter) != num:
            raise ValueError("If `weights` are provided, number of `filter` values must match.")

        # Filter `xvals`
        if filter[0] is not None:
            inds = zmath.comparison_filter(xvals, filter[0], inds=True)
            xvals = xvals[inds]
            yvals = yvals[inds]
            if weights is not None:
                weights = weights[inds]
        # Filter `yvals`
        if filter[1] is not None:
            inds = zmath.comparison_filter(yvals, filter[1], inds=True)
            xvals = xvals[inds]
            yvals = yvals[inds]
            if weights is not None:
                weights = weights[inds]

        if weights is not None and filter[2] is not None:
            inds = zmath.comparison_filter(yvals, filter[2], inds=True)
            xvals = xvals[inds]
            yvals = yvals[inds]
            weights = weights[inds]

    # Create and initializae figure and axes
    fig, prax, xpax, ypax, cbax = _constructFigure(fig, xproj, yproj, hratio, wratio, pad,
                                                   scale, histScale, labels, cbar,
                                                   left, bottom, right, top, fs=fs)

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

    # Make sure bins look okay
    for arr, name in zip([xbins, ybins], ['xbins', 'ybins']):
        delta = np.diff(arr)
        if np.any(~np.isfinite(delta) | (delta == 0.0)):
            raise ValueError("Error constructing `{}` = {}, delta = {}".format(name, arr, delta))

    # Calculate Histograms
    # --------------------
    #    2D
    hist_2d, xedges_2d, yedges_2d, binnums = sp.stats.binned_statistic_2d(
        xvals, yvals, weights, statistic=statistic, bins=[xbins, ybins])
    hist_2d = np.nan_to_num(hist_2d)
    #    X-projection (ignore Y)
    hist_xp, edges_xp, bins_xp = sp.stats.binned_statistic(
        xvals, weights, statistic=statistic, bins=xbins)
    #    Y-projection (ignore X)
    hist_yp, edges_yp, bins_yp = sp.stats.binned_statistic(
        yvals, weights, statistic=statistic, bins=ybins)

    # Calculate Extrema - Preserve input extrema if given, otherwise calculate
    extrema = _set_extrema(extrema, [hist_2d, hist_xp, hist_yp], filter=filter[2])

    # Create scalar-mappable if needed
    if smap is None:
        smap = plot_core.colormap(extrema, cmap=cmap, scale=histScale)

    # Plot Histograms and Projections
    # -------------------------------
    # Plot 2D Histogram
    if type_hist:
        counts = None
        # If we should overlay strings labeling the num values in each bin, calculate those `counts`
        if write_counts:
            counts, xedges_2d, yedges_2d, binnums = sp.stats.binned_statistic_2d(
                xvals, yvals, weights, statistic='count', bins=[xbins, ybins])

        pcm, smap = plot2DHist(prax, xedges_2d, yedges_2d, hist_2d, cscale=histScale, cbax=cbax,
                               labels=labels, counts=counts, cmap=cmap, smap=smap, extrema=extrema)

        # Colors
        # X-projection
        colhist_xp = np.array(hist_xp)
        # Enforce positive values for colors in log-plots.
        if smap.log:
            tmin, tmax = zmath.minmax(colhist_xp, filter='g')
            colhist_xp = np.maximum(colhist_xp, tmin)
        colors_xp = smap.to_rgba(colhist_xp)
        colors_yp = smap.to_rgba(hist_yp)

    # Scatter Plot
    else:
        colors = smap.to_rgba(weights)
        prax.scatter(xvals, yvals, c=colors, alpha=alpha)

        cbar = plt.colorbar(smap, cax=cbax)
        cbar.set_label(cblabel, fontsize=fs)
        cbar.ax.tick_params(labelsize=fs)

        # Make projection colors all grey
        colors_xp = '0.8'
        colors_yp = '0.8'

    # Plot projection of the x-axis (i.e. ignore 'y')
    if xpax:
        islog = scale[0].startswith('log')
        # extrema_y = [zmath.floor_log(extrema[0]), zmath.ceil_log(extrema[1])]
        extrema_y = zmath.minmax(extrema, round=0)

        xpax.bar(edges_xp[:-1], hist_xp, color=colors_xp, log=islog, width=np.diff(edges_xp))
        #     set tick-labels to the top
        plt.setp(xpax.get_yticklabels(), fontsize=fs)
        xpax.xaxis.tick_top()
        #     set bounds to bin edges
        plot_core.setLim(xpax, 'x', data=xedges_2d)
        xpax.set_ylim(extrema_y)

    # Plot projection of the y-axis (i.e. ignore 'x')
    if ypax:
        islog = plot_core._scale_to_log_flag(histScale)
        ypax.barh(edges_yp[:-1], hist_yp, color=colors_yp, log=islog, height=np.diff(edges_yp))
        #     set tick-labels to the top
        plt.setp(ypax.get_yticklabels(), fontsize=fs, rotation=90)
        ypax.yaxis.tick_right()
        #     set bounds to bin edges
        plot_core.setLim(ypax, 'y', data=yedges_2d)
        try:
            ypax.locator_params(axis='x', tight=True, nbins=4)
        except:
            ypax.locator_params(axis='x', tight=True)

    prax.set(xlim=zmath.minmax(xedges_2d), ylim=zmath.minmax(yedges_2d))

    return fig


def plot2DHist(ax, xvals, yvals, hist,
               cax=None, cbax=None, cscale='log', cmap=None, smap=None, extrema=None,
               contours=None, clabel={}, fs=12, rasterized=True,
               title=None, labels=None, counts=None, **kwargs):
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
    hist : (N,M) ndarray of scalars
        Grid of data-points to plot.
    cax : `matplotlib.axes.Axes` object
        Axes object on which to add a colorbar.
        See the `cax` parameter of `plt.colorbar`.
    cbax : `matplotlib.axes.Axes` object(s)
        Axes object from which to make space for a colorbar axis.
        See the `ax` parameter of `plt.colorbar`.
    cscale : str
        Scale to use for the colormap {'linear', 'log'}.
        Overridden if `smap` is provided.
    cmap : ``matplotlib.colors.Colormap`` object
        Matplotlib colormap to use for coloring histogram.
        Overridden if `smap` is provided.
    fs : int
        Fontsize specification.
    title : str or `None`
        Title to add to top of axes.
    smap : `matplotlib.cm.ScalarMappable` object or `None`
        A scalar-mappable object to use for colormaps, or `None` for one to be created.
    extrema : (2,) array_like of scalars
        Minimum and maximum values for colormap scaling.
    labels : (2,) or (3,) array_like of strings
        The first two string are the 'x' and 'y' axis labels respectively.  If a third string is
        provided it is used as the colorbar label.
    counts : (N,M) ndarray of int or `None`
        Number of elements in each bin if overlaid-text is desired.

    Returns
    -------
    pcm : `matplotlib.collections.QuadMesh` object
        The resulting plotted object, returned by ``ax.pcolormesh``.
    smap : `matplotlib.cm.ScalarMappable` object
        Colormap and color-scaling information.  See: ``zcode.plot.plot_core.colormap``.

    """
    cblab = 'Counts'
    xgrid, ygrid = np.meshgrid(xvals, yvals)
    hist = np.asarray(hist)
    if plot_core._scale_to_log_flag(cscale): filter = 'g'
    else:                                    filter = None
    extrema = _set_extrema(extrema, hist, filter=filter)

    if labels is not None:
        if np.size(labels) >= 2:
            ax.set_xlabel(labels[0], size=fs)
            ax.set_ylabel(labels[1], size=fs)
        if np.size(labels) > 2:
            cblab = labels[2]

    # Create scalar-mappable if needed
    if smap is None:
        smap = plot_core.colormap(extrema, cmap=cmap, scale=cscale)

    # Plot
    pcm = ax.pcolormesh(xgrid, ygrid, hist.T, norm=smap.norm, cmap=smap.cmap, linewidth=0,
                        rasterized=rasterized, vmin=smap.norm.vmin, vmax=smap.norm.vmax, **kwargs)
    pcm.set_edgecolor('face')

    # Add color bar
    if cbax is not None or cax is not None:
        if cbax is not None:
            cbar = plt.colorbar(smap, cax=cbax)
        else:
            cbar = plt.colorbar(smap, ax=cax)
        cbar.set_label(cblab, fontsize=fs)
        cbar.ax.tick_params(labelsize=fs)

    if fs is not None:
        ax.tick_params(labelsize=fs)

    if title is not None:
        ax.set_title(title, size=fs)

    # Add counts overlay
    if counts is not None:
        counts = np.asarray(counts).astype(int)
        # Make sure sizes are correct
        if counts.shape != hist.shape:
            raise ValueError("shape of `counts` ({}) must match `hist` ({})".format(
                counts.shape, hist.shape))

        # Remember these are transposes
        for ii in xrange(xgrid.shape[0] - 1):
            for jj in xrange(xgrid.shape[1] - 1):
                xx = np.sqrt(xgrid[jj, ii] * xgrid[jj, ii+1])
                yy = np.sqrt(ygrid[jj, ii] * ygrid[jj+1, ii])
                ax.text(xx, yy, "{:d}".format(counts.T[jj, ii]), ha='center', va='center',
                        fontsize=8, bbox=dict(facecolor='white', alpha=0.2, edgecolor='none'))

    # Add contour lines
    if contours is not None:
        if isinstance(contours, bool) and contours:
            levels = None
        else:
            levels = np.array(contours)

        xg, yg = np.meshgrid(zmath.midpoints(xvals, log=True), zmath.midpoints(yvals, log=True))
        cs = ax.contour(xg, yg, hist[:-1, :-1].T, colors='0.25', norm=smap.norm,
                        levels=levels, linewidths=4.0, antialiased=True)
        ax.contour(xg, yg, hist[:-1, :-1].T, cmap=smap.cmap, norm=smap.norm,
                   levels=levels, linewidths=1.5, antialiased=True)
        # plt.clabel(cs, inline=1, fontsize=fs, fmt='%.0e')
        if levels is not None and clabel is not None:
            plt.clabel(cs, inline=1, **clabel)

    plot_core.setLim(ax, 'x', data=xvals)
    plot_core.setLim(ax, 'y', data=yvals)
    return pcm, smap


def _constructFigure(fig, xproj, yproj, hratio, wratio, pad, scale, histScale, labels, cbar,
                     left, bottom, right, top, fs=12):
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
    if not fig: fig = plt.figure()

    xpax = ypax = cbax = None

    # Determine usable space and axes sizes
    useable = [right-left, top-bottom]
    if cbar:
        useable[0] -= _CB_WID + _CB_WPAD

    if yproj:
        prim_wid = useable[0]*wratio
        ypro_wid = useable[0]*(1.0-wratio)
    else:
        prim_wid = useable[0]

    if xproj:
        prim_hit = useable[1]*hratio
        xpro_hit = useable[1]*(1.0-hratio)
    else:
        prim_hit = useable[1]

    # Create primary axes, at bottom left
    #    d
    prax = fig.add_axes([left, bottom, prim_wid, prim_hit])
    prax.set(xscale=scale[0], yscale=scale[1], xlabel=labels[0], ylabel=labels[1])
    prax.tick_params(axis='both', which='major', labelsize=fs)
    plot_core.setGrid(prax, False)

    if len(labels) > 2: histLab = labels[2]
    else:               histLab = 'Counts'

    # Add x-projection axes on top-left
    if xproj:
        xpax = fig.add_axes([left, bottom+prim_hit+pad, prim_wid, xpro_hit-pad])
        xpax.set(xscale=scale[0], yscale=histScale, ylabel=histLab, xlabel=labels[0])
        xpax.xaxis.set_label_position('top')
        xpax.tick_params(axis='both', which='major', labelsize=fs)
        plot_core.setGrid(xpax, True, axis='y')

    # Add y-projection axes on bottom-right
    if yproj:
        ypax = fig.add_axes([left+prim_wid+pad, bottom, ypro_wid-pad, prim_hit])
        ypax.set(yscale=scale[1], xscale=histScale, xlabel=histLab, ylabel=labels[1])
        ypax.yaxis.set_label_position('right')
        ypax.tick_params(axis='both', which='major', labelsize=fs)
        plot_core.setGrid(ypax, True, axis='x')

    # Add color-bar axes on the right
    if cbar:
        cbar_left = left + prim_wid + _CB_WPAD
        if yproj: cbar_left += ypro_wid
        cbax = fig.add_axes([cbar_left, bottom, _CB_WID, prim_hit])

    return fig, prax, xpax, ypax, cbax


def _set_extrema(extrema, vals, filter=None):
    _extr = None
    for vv in vals:
        _extr = zmath.minmax(vv, filter=filter, prev=_extr, stretch=0.05)
    if extrema is None: new_extr = _extr
    else:               new_extr = np.asarray(extrema)
    if new_extr[0] is None: new_extr[0] = _extr[0]
    if new_extr[1] is None: new_extr[1] = _extr[1]
    new_extr = new_extr.astype(np.float64)
    return new_extr
