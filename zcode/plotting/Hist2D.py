"""

Functions
---------
-    plot2DHist              - Plot a 2D histogram
-    plot2DHistProj          - Plot a 2D histogram, with projected 1D histograms

"""
import numpy as np
import matplotlib.pyplot as plt

import zcode.Math as zmath
import zcode.Plotting as zplot

__all__ = ['plot2DHist', 'plot2DHistProj']

_LEFT = 0.1
_RIGHT = 0.05
_BOTTOM = 0.1
_TOP = 0.1


def _constructFigure(fig, figsize, xproj, yproj, hratio, wratio):
    assert 0.0 <= hratio <= 1.0, "`hratio` must be between [0.0,1.0]!"
    assert 0.0 <= wratio <= 1.0, "`wratio` must be between [0.0,1.0]!"

    # Create figure if needed
    if(not fig): fig = plt.figure(figsize=figsize)

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
    count = 1
    # Add x-projection axes on top-left
    if(xproj):
        fig.add_axes([_LEFT, _BOTTOM+prim_hit, prim_wid, xpro_hit])
        xkey = count
        count += 1

    # Add y-projection axes on bottom-right
    if(yproj):
        fig.add_axes([_LEFT+prim_wid, _BOTTOM, ypro_wid, prim_hit])
        ykey = count

    return fig, xkey, ykey


def plot2DHistProj(xvals, yvals, bins=10, fig=None, figsize=[16, 12], xproj=True, yproj=True,
                   hratio=0.7, wratio=0.7, fs=12, interp='nearest'):
    """
    """
    fig, xk, yk = _constructFigure(fig, xproj, yproj, hratio, wratio)
    axes = fig.axes

    # `bins` is a single scalar value -- apply to both
    if(np.ndim(bins) == 0):
        xbins = bins
        ybins = bins
    else:
        # `bins` is a pair of bin specifications, separate and apply
        if(len(bins) == 2):
            xbins = bins[0]
            ybins = bins[1]
        # `bins` is a single array -- apply to both
        elif(len(bins) > 2):
            xbins = bins
            ybins = bins
        # unrecognized option -- error
        else:
            raise ValueError("Unrecognized shape of ``bins`` = %s" % (str(np.shape(bins))))

    axes[0].set_xticklabels(())
    axes[0].set_yticklabels(())
    hist, xedges, yedges = np.histogram2d(xvals, yvals, bins=bins)
    # extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    hist_im = axes[0].imshow(hist.T, interpolation=interp, origin='lower')
    # cb = plt.colorbar(hist_im, orientation='horizontal', cax=axCB)
    if(xproj):
        axes[xk].hist(xvals, bins=xbins, color='0.5')
        plt.setp(axes[xk].get_yticklabels(), fontsize=fs)

    if(yproj):
        axes[yk].hist(yvals, bins=ybins, orientation='horizontal', color='0.5')
        plt.setp(axes[yk].get_yticklabels(), fontsize=fs)

    return fig
# } def plot2DHistProj


def plot2DHist(ax, xvals, yvals, hist, cbax=None, cscale='log', cmap=plt.cm.jet, fs=12,
               extrema=None, **kwargs):
    """
    Plot the given 2D histogram of data.

    Use with (e.g.) ``numpy.histogram2d``,

    Arguments
    ---------
        ax     <obj>      : <matplotlib.axes.Axes> object on which to plot.
        xvals  <flt>[N]   : positions of x values, assumed to be the same for all rows of ``data``


        cbax   <obj>      : <matplotlib.axes.Axes> object on which to add a colorbar.
        cscale <str>      : scale to use for the colormap {'linear','log'}
        cmap   <obj>      : <matplotlib.colors.Colormap> object to use as colormap

    """

    xgrid, ygrid = np.meshgrid(xvals, yvals)

    if(extrema is None): extrema = zmath.minmax(hist, nonzero=(cscale == 'log'))

    # Plot
    smap = zplot.colormap(extrema, cmap=cmap, scale=cscale)
    pcm = ax.pcolormesh(xgrid, ygrid, hist.T, norm=smap.norm, cmap=smap.cmap,  # edgecolors='face',
                        **kwargs)

    # Add color bar
    if(cbax is not None):
        cbar = plt.colorbar(smap, cax=cbax)
        cbar.set_label('Counts', fontsize=fs)
        cbar.ax.tick_params(labelsize=fs)

    zplot.set_lim(ax, 'x', data=xvals)
    zplot.set_lim(ax, 'y', data=yvals)

    return pcm
# } def plot2DHist
