"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import matplotlib.pyplot as plt
import numbers

import zcode.math as zmath
from . import plot_core as zplot

__all__ = ['plotCorrelationGrid']


_LEFT = 0.06
_RIGHT = 0.95
_TOP = 0.95
_BOT = 0.05


def plotCorrelationGrid(data, figure=None, style='scatter', confidence=True, contours=True,
                        pars_scales=None, hist_scales=None, hist_bins=None, names=None, fs=12):
    """
    Plot a grid of correlation graphs, showing histograms of arrays and correlations between pairs.

    Arguments
    ---------
        data <scalar>[N][M]        : ``N`` different parameters, with ``M`` values each

        figure      <obj>          : ``matplotlib.figure.Figure`` object on which to plot
        style       <str>          : what style of correlation plots to make
                                     - 'scatter'

        confidence  <bool>         : Add confidence intervals to histogram plots
        contours    <bool>         : Add contour lines to correlation plots
        pars_scales  <scalar>([N]) : What scale to use for all (or each) parameter {'lin', 'log'}
        hist_scales <scalar>([N])  : What y-axis scale to use for all (or each) histogram
        hist_bins   <scalar>([N])  : Number of bins for all (or each) histogram


    Returns
    -------
        figure <obj>      : ``matplotlib.figure.Figure`` object
        axes   <obj>[N,N] : array of ``matplotlib.axes`` objects

    """

    npars = len(data)

    # Set default scales for each parameter
    if(pars_scales is None):            pars_scales = ['linear']*npars
    elif(isinstance(pars_scales, str)): pars_scales = [pars_scales]*npars

    # Set default scales for each histogram (counts)
    if(hist_scales is None):            hist_scales = ['linear']*npars
    elif(isinstance(hist_scales, str)): hist_scales = [hist_scales]*npars

    # Convert scaling strings to appropriate formats
    for ii in range(npars):
        if(pars_scales[ii].startswith('lin')): pars_scales[ii] = 'linear'
        elif(pars_scales[ii].startswith('log')): pars_scales[ii] = 'log'
        if(hist_scales[ii].startswith('lin')): hist_scales[ii] = 'linear'
        elif(hist_scales[ii].startswith('log')): hist_scales[ii] = 'log'

    # Set default bins
    if(hist_bins is None):                         hist_bins = [40]*npars
    elif(isinstance(hist_bins, numbers.Integral)): hist_bins = [hist_bins]*npars

    # Setup Figure and Axes
    # ---------------------
    #     Create Figure
    if(figure is None): figure = plt.figure()

    # Axes are already on figure
    if(len(figure.axes) > 0):
        # Make sure the number of axes is correct
        if(len(figure.axes) != npars*npars):
            raise RuntimeError("``figure`` axes must be {0:d}x{0:d}!".format(npars))

    # Create axes
    else:
        # Divide figure evenly with padding
        dx = (_RIGHT-_LEFT)/npars
        dy = (_TOP-_BOT)/npars

        # Rows
        for ii in range(npars):
            # Columns
            for jj in range(npars):
                ax = figure.add_axes([_LEFT+jj*dx, _TOP-(ii+1)*dy, dx, dy])
                # Make upper-right half of figure invisible
                if(jj > ii):
                    ax.set_visible(False)
                    continue

    axes = np.array(figure.axes)
    # Reshape to grid for convenience
    axes = axes.reshape(npars, npars)

    # Plot Correlations and Histograms
    # --------------------------------
    lims = []
    for ii in range(npars):
        lims.append(zmath.minmax(data[ii]))

        for jj in range(npars):
            if(jj > ii): continue

            # Histograms
            if(ii == jj):
                zplot.plotHistBars(axes[ii, jj], data[ii], bins=hist_bins[ii],
                                   scalex=pars_scales[ii], conf=True)

            # Correlations
            else:
                if(style == 'scatter'):
                    zplot.plotScatter(axes[ii, jj], data[jj], data[ii],
                                      scalex=pars_scales[jj], scaley=pars_scales[ii], cont=contours)
                else:
                    raise RuntimeError("``style`` '%s' is not implemented!" % (style))

    # Configure Axes
    _config_axes(axes, lims, pars_scales, hist_scales, names, fs)

    return figure, axes


def _config_axes(axes, lims, par_scales, hist_scales, names, fs):
    shp = np.shape(axes)
    assert len(shp) == 2 and shp[0] == shp[1], "``axes`` must have shape NxN!"
    npars = shp[0]

    for ii in range(npars):
        for jj in range(npars):
            if(jj > ii): continue

            ax = axes[ii, jj]
            for xy in ['x', 'y']: ax.tick_params(axis=xy, which='both', labelsize=fs)
            zplot.set_grid(ax, True)

            # Setup Scales and Limits
            # -----------------------
            ax.set_xscale(par_scales[jj])
            if(ii == jj):
                ax.set_yscale(hist_scales[ii], nonposy='clip')
                if(hist_scales[ii].startswith('log')): zplot.setLim(ax, lo=0.8)
                ax.set_xlim(lims[ii])
            else:
                ax.set_yscale(par_scales[ii])
                ax.set_ylim(lims[ii])
                ax.set_xlim(lims[jj])

            # Add axis labels
            if(names is not None):
                if(jj == 0):       ax.set_ylabel(names[ii], fontsize=fs)
                if(ii == npars-1): ax.set_xlabel(names[jj], fontsize=fs)

            # Setup Ticks
            # -----------

            # Move histogram ticks to right-side
            if(ii == jj):
                ax.yaxis.tick_right()
            # Remove y-ticks from non-left-most axes
            elif(jj > 0):
                ax.set_yticklabels(['' for tick in ax.get_yticks()])
            # Remove overlapping ticks
            else:

                if(ii > 0):
                    ticks = ax.yaxis.get_major_ticks()
                    ticks[-1].label1.set_visible(False)

            # Remove x-ticks from non-bottom axes
            if(ii < npars-1):
                ax.set_xticklabels(['' for tick in ax.get_yticks()])
            # Remove overlapping ticks
            else:

                if(jj < npars-1):
                    ticks = ax.xaxis.get_major_ticks()
                    ticks[-1].label1.set_visible(False)

    return

# } def _config_axes
