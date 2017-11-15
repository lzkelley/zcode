"""

-   plot_hist_line         - Plot a histogram line.
-   plot_segmented_line    - Draw a line segment by segment.
-   plot_scatter           - Draw a scatter plot.
-   plot_hist_bars         - Plot a histogram bar graph.
-   plot_conf_fill         - Draw a median line and (set of) confidence interval(s).

"""
import numbers

import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np

from zcode import math as zmath
from zcode import utils

from . plot_const import COL_CORR, LW_CONF, LW_OUTLINE
from . plot_core import colormap

__all__ = [
    "plot_hist_line", "plot_segmented_line", "plot_scatter", "plot_hist_bars", "plot_conf_fill",
    # Deprecated
    "plotHistLine", "plotSegmentedLine", "plotScatter", "plotHistBars", "plotConfFill"
]


def plot_hist_line(ax, edges, hist, yerr=None, nonzero=False, positive=False, extend=None,
                   fill=None, invert=False, **kwargs):
    """Given bin edges and histogram-like values, plot a histogram.

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
    if kwargs.get('color') is not None:
        col = kwargs.get('color')
    elif kwargs.get('c') is not None:
        col = kwargs.get('c')

    yerr_fmt = '+'

    # Extend bin edges if needed
    if len(edges) != len(hist)+1:
        if extend == 'left':
            edges = np.concatenate([[zmath.extend(edges)[0]], edges])
        elif extend == 'right':
            edges = np.concatenate([edges, [zmath.extend(edges)[1]]])
        else:
            raise RuntimeError("``edges`` must be longer than ``hist``, or ``extend`` given")

    # Construct plot points to manually create a step-plot
    xval, yval = _hist_line(edges, hist)

    # Select nonzero values
    if nonzero:
        xval = np.ma.masked_where(yval == 0.0, xval)
        yval = np.ma.masked_where(yval == 0.0, yval)

    # Select positive values
    if positive:
        xval = np.ma.masked_where(yval < 0.0, xval)
        yval = np.ma.masked_where(yval < 0.0, yval)

    if invert:
        temp = np.array(xval)
        xval = yval
        yval = temp

    # Plot Histogram
    line, = ax.plot(xval, yval, **kwargs)

    # Add yerror-bars
    if yerr is not None:
        xmid = zmath.midpoints(edges)

        if nonzero:
            inds = (hist != 0.0)
            ax.errorbar(xmid[inds], hist[inds], yerr=yerr[inds], fmt=yerr_fmt, ecolor=col)
        else:
            ax.errorbar(xmid,       hist,       yerr=yerr,       fmt=yerr_fmt, ecolor=col)

    # Add a fill region
    if fill is not None:
        ylim = ax.get_ylim()
        if type(fill) == dict:
            filldict = fill
        else:
            filldict = dict()
        ax.fill_between(xval, yval, 0.1*ylim[0], **filldict)
        ax.set_ylim(ylim)

    return line


def plot_segmented_line(ax, xx, yy, zz=None, cmap=plt.cm.jet, norm=[0.0, 1.0], lw=3.0, alpha=1.0):
    """Draw a line segment by segment.
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb

    Plot a colored line with coordinates xx and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Get the minimum and maximum of ``norm``
    norm = zmath.minmax(norm)
    # conver to normalization
    norm = plt.Normalize(norm[0], norm[1])

    if zz is None:
        zz = np.linspace(norm.vmin, norm.vmax, num=len(xx))
    else:
        zz = np.asarray(zz)

    segments = _make_segments(xx, yy)
    lc = mpl.collections.LineCollection(segments, array=zz, cmap=cmap, norm=norm,
                                        linewidth=lw, alpha=alpha)

    ax.add_collection(lc)

    return lc


def plot_scatter(ax, xx, yy, scalex='log', scaley='log',
                 size=None, cont=False, color=None, alpha=None, **kwargs):
    """Draw a scatter plot.
    """
    COL = COL_CORR

    if size is None: size = [30, 6]
    if color is None: color = ['0.25', COL]
    if alpha is None: alpha = [0.5, 0.8]

    ax.scatter(xx, yy, s=size[0], color=color[0], alpha=alpha[0], **kwargs)
    pnts = ax.scatter(xx, yy, s=size[1], color=color[1], alpha=alpha[1], **kwargs)

    # Add Contours
    if cont:
        NUM = 4
        CMAP = 'jet'
        res = 0.3*np.sqrt(len(xx))
        res = np.max([1, res])

        smap = colormap(NUM, CMAP, scale='lin')
        cols = [smap.to_rgba(it) for it in range(NUM)]

        xi = zmath.spacing(xx, scalex, num=res)
        yi = zmath.spacing(yy, scaley, num=res)
        hist, xedges, yedges = np.histogram2d(xx, yy, (xi, yi))
        gx, gy = np.meshgrid(xedges[1:], yedges[1:])
        ax.contour(gx, gy, hist.T, NUM, colors=cols)

    return pnts


def plot_hist_bars(ax, xx, bins=20, scalex='log', scaley='log', conf=True, **kwargs):
    """Plot a histogram bar graph.

    NOTE: For log-y, make sure `yscale` is either not manually set, or include `nonposy='clip'`

    Arguments
    ---------
    ax : ``matplotlib.axes.Axes`` object,
        Axes on which to plot.
    xx : (N,) array_like scalars,
        Values to be histogrammed.
    bins : int or array_like,
        Either the number of bins for bin-edges to be automatically generated, or the bin-edges
        themselves.
    ...

    """
    HIST_ALPHA = 0.75
    CONF_ALPHA = 0.5

    CONF_INTS = [0.95, 0.68]
    CONF_COLS = ['green', 'orange']

    if scaley.startswith('log'):
        logy = True
    else:
        logy = False

    if 'color' not in kwargs and 'c' not in kwargs:
        kwargs['color'] = COL_CORR
    if 'alpha' not in kwargs:
        kwargs['alpha'] = HIST_ALPHA
    if 'rwidth' not in kwargs:
        kwargs['rwidth'] = 0.8
    if 'zorder' not in kwargs:
        kwargs['zorder'] = 100

    # Add Confidence intervals
    if conf:
        med, cints = zmath.confidenceIntervals(xx, ci=CONF_INTS)
        ax.axvline(med, color='red', ls='--', lw=LW_CONF, zorder=101)
        # Add average
        ave = np.average(xx)
        ax.axvline(ave, color='red', ls=':', lw=LW_CONF, zorder=101)
        for ii, (vals, col) in enumerate(zip(cints, CONF_COLS)):
            ax.axvspan(*vals, color=col, alpha=CONF_ALPHA)

    # Create a given number of log-spaced bins
    #     If not log-spaced, then `ax.hist` will do the same
    if isinstance(bins, numbers.Integral) and scalex.startswith('log'):
        bins = zmath.spacing(xx, num=bins, scale='log')

    cnts, bins, bars = ax.hist(xx, bins, histtype='bar', log=logy, **kwargs)

    # Dont let lower y-lim be less than 0.8 with log-scaling
    if scaley.startswith('log'):
        # setLim(ax, 'y', lo=0.8, at='least')   <=== This isn't working for some reason!  FIX
        ylim = np.array(ax.get_ylim())
        if ylim[0] < 0.8:
            ylim[0] = 0.8
            ax.set_ylim(ylim)

    return bars


def plot_conf_fill(ax, rads, med, conf, color='firebrick', fillalpha=0.5, lw=1.0, linealpha=0.8,
                   filter=None, outline='0.5', edges=True, floor=None, ceil=None, dashes=None,
                   lw_edges=None,
                   **kwargs):
    """Draw a median line and (set of) confidence interval(s).

    The `med` and `conf` values can be obtained from `numpy.percentile` and or
    `zcode.math.confidenceIntervals`.

    Arguments
    ---------
    ax : `matplotlib.axes.Axes` object,
        Axes on which to plot.
    rads : (N,) array_like scalars,
        Radii corresponding to each median and confidence value.
    med : (N,) array_like scalars,
        Median values to plot as line.
    conf : (N,[M,]2,) array_like scalars,
        Confidence intervals to plot as shaded regions.  There can be `M` different confidence
        intervals, each with an upper and lower value.  Shape of `(N,2,)` is also allowed for a
        single confidence interval.
    color : `matplotlib` color spec,
        Color for median lines and shaded region.
    fillalpha : float or (M,) array_like of floats,
        Alpha (opacity) specification for fill regions; each element must be ``{0.0, 1.0}``.
        If ``(M,)`` are given, one is used for each confidence-interval.  Otherwise, for
        confidence interval `i`, the alpha used is ``fillalpha^{i+1}``.
    lw : float,
        Line-weight used specifically for the median line (not the filled-regions).
    filter : str or `None`
        Apply a relative-to-zero filter to the y-data before plotting.
    outline : str or `None`.
        Draw a `outline` colored line behind the median line to make it easier to see.
        If `None`, no outline is drawn.
    edges : bool
        Add lines along the edges of each confidence interval.
    floor : array_like of float or `None`
        Set the minimum value for confidence intervals to this value.
    ceil : array_like of float or `None`
        Set the maximmum value for confidence intervals to be this value.
    dashes :
    **kwargs : additional key-value pairs,
        Passed to `matplotlib.pyplot.fill_between` controlling `matplotlib.patches.Polygon`
        properties.  These are included in the `line_patch` objects, but *not* the `conf_patches`.

    Returns
    -------
    line_patch : `matplotlib.patches.Patch`,
        Composite patch of median line and shaded region patch (for use on legend).
    conf_patches : (M,) list of `matplotlib.patches.Patch`,
        A patch for each confidence inteval (for use on legend).

    """
    conf = np.atleast_2d(conf)
    if conf.shape[-1] != 2:
        raise ValueError("Last dimension of `conf` must be 2!")

    # `conf` has shape ``(num-rads, num-conf-ints, 2)``
    if conf.ndim == 2:
        conf = conf.reshape(len(rads), 1, 2)
    elif conf.ndim != 3:
        raise ValueError("`conf` must be 2 or 3 dimensions!")

    if filter is not None:
        filter = zmath._comparisonFunction(filter)

    if lw_edges is None:
        lw_edges = 0.5 * lw

    num_conf = np.shape(conf)[-2]
    conf_patches = []
    # Iterate over confidence intervals
    _pp = None
    for jj in range(num_conf):
        # Set fill-opacity
        if np.size(fillalpha) == num_conf:
            falph = fillalpha
        else:
            falph = np.power(fillalpha, jj+1)

        xx = np.array(rads)
        ylo = np.array(conf[:, jj, 0])
        yhi = np.array(conf[:, jj, 1])
        if floor is not None:
            ylo = np.maximum(ylo, floor)
        if ceil is not None:
            yhi = np.minimum(yhi, ceil)

        if filter:
            ylo = np.ma.masked_where(~filter(ylo, 0.0), ylo)
            yhi = np.ma.masked_where(~filter(yhi, 0.0), yhi)

        # Fill between confidence intervals
        pp = ax.fill_between(xx, ylo, yhi, alpha=falph, facecolor=color, **kwargs)
        conf_patches.append(pp)

        # Plot edges of confidence intervals
        if edges:
            ax.plot(rads, ylo, color=color, alpha=0.5*linealpha, lw=lw_edges)
            ax.plot(rads, yhi, color=color, alpha=0.5*linealpha, lw=lw_edges)

        # Create dummy-patch for the median-line and fill-color, for a legend
        if jj == 0:
            # pp = ax.fill(np.nan, np.nan, facecolor=color, alpha=falph, **kwargs)
            # Create overlay of lines and patches
            _pp = pp
            # line_patch = (_pp, ll)

    # Plot Median Line
    #    Plot black outline to improve contrast
    if outline is not None:
        oo, = ax.plot(rads, med, '-', color=outline, lw=2*lw, alpha=LW_OUTLINE)
        if dashes is not None:
            oo.set_dashes(tuple(dashes))

    ll, = ax.plot(rads, med, '-', color=color, lw=lw, alpha=linealpha)
    if dashes is not None:
        ll.set_dashes(tuple(dashes))

    line_patch = (_pp, ll)

    return line_patch, conf_patches


def _hist_line(edges, hist):
    xval = np.hstack([[edges[jj], edges[jj+1]] for jj in range(len(edges)-1)])
    yval = np.hstack([[hh, hh] for hh in hist])
    return xval, yval


def _make_segments(x, y):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb

    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   [numlines, (points per line), 2 (x and y)] array
    """
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


# === Deprecations ===


def plotHistLine(*args, **kwargs):
    utils.dep_warn("plotHistLine", newname="plot_hist_line")
    return plot_hist_line(*args, **kwargs)


def plotSegmentedLine(*args, **kwargs):
    utils.dep_warn("plotSegmentedLine", newname="plot_segmented_line")
    return plot_segmented_line(*args, **kwargs)


def plotScatter(*args, **kwargs):
    utils.dep_warn("plotScatter", newname="plot_scatter")
    return plot_scatter(*args, **kwargs)


def plotHistBars(*args, **kwargs):
    utils.dep_warn("plotHistBars", newname="plot_hist_bars")
    return plot_hist_bars(*args, **kwargs)


def plotConfFill(*args, **kwargs):
    utils.dep_warn("plotConfFill", newname="plot_conf_fill")
    return plot_conf_fill(*args, **kwargs)
