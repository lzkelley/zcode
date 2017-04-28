CHANGES
=======

Future / To-Do
--------------
-   General
    +   Implement tests for all modules and functions.
    +   Update all situations to use 'filter' instead of 'positive'/'nonzero'
    +   More tests.  For plotting stuff, add tests just to make sure nothing breaks.
    +   Setup automatic testing (e.g. nightly?)
-   math/
    +   math_core.py
        -   `spacing`
            +   Add `endpoint` keyword argument to decide whether end-point(s) are included
                in returned range.
        -   `around`
            +   Does this work correctly for negative decimal values??
            +   Implement 'dir' relative to zero --- i.e. round up/down the absolute values.
-   plot/
    +   Hist2D.py
        -   Add ability to plot central axes as scatter plot, with projected histograms
            (instead of just the central 2D histogram).
    +   plot_core.py
        -   Finish 'set_ticks' method.


Current
-------
-  `inout/`
    -   `inout_core.py`
        -   `check_path()` <== `checkPath` [DEPRECATION]
        -   `getFileSize()` [DELETED]
            -   Use `get_file_size()` instead.
        -   `modify_exists()`
            -   If, for some reason, the new filename already exists, raise a warning and then bootstrap to modify the filename again.  Previously the code would raise an error.
        -   `modify_filename()` <== `modifyFilename` [DEPRECATION]
    -   `log.py`
        -   Add method `after()` to logger objects which report a message and duration for execution.
        -   Add method `frac()` to logger objects which report a fraction.
        -   Changed parameters for logging methods to use underscores instead of camel-case.
        -   `get_logger()` <== `getLogger` [DEPRECATION]
        -   `default_logger()` <== `defaultLogger` [DEPRECATION]
        -   Added docstrings to `after()`, `raise_error()`, and `copy()` added-on methods.
-   `plot/`
    -   `plot_core.py`
        -   `text()`
            -   Add a `shift` argument which allows for adjusting the `(x,y)` position of the text more dynamically.
        -   `_loc_str_to_pars()`
            -   Check the location specifier for validity.
        -   `set_grid()` <== `setGrid` [DEPRECATION]
        -   `set_lim()` <== `setLim` [DEPRECATION]
            
-   `math/`
    -   `statistic.py`
        -   `stats_str`
            -   Re-enabled the `label` argument for backwards compatibility.  If used, a warning is raised.  But it works.
    -   `math_core.py`
        -   `datetime_to_decimal_year()` [new-function]
            -   New function to convert from a datetime object (or string datetime specification) to a decimal year.
            -   Added precision down to milliseconds.
        -   `str_array()`
            -   Changed the arguments to this function to use a single `sides` parameter which encodes information about both the beginning and end.
            -   Improved the function to properly handle the number of elements at the end, and what to do if the number of requested elements equals or exceeds the array length.
            -   Added tests to `tests.test_math_core.TestMathCore.test_str_array()`.


[0.0.9] - 2017/03/07
--------------------
-   inout/
    +   inout_core.py
        -   `npzToDict`
            +   BUG: issue loading npz across python2-python3 transition.  Attempt to resolve.
        -   `str_format_dict` [new-function]
            -   New function to pretty-print a dictionary object into a string (uses `json`).
        -   `getFileSize` ==> `get_file_size` [deprecation]
            -   Also improve behavior to accept single or list of filenames.
        -   `getProgressBar` [DELETED]
            -   Should use `tqdm` functions instead.
        -  `par_dir` [new-function]
            -   !!NOTE: not sure if this is a good one... commented out for now!!
            -   Method which returns the parent directory of the given path.
        -  `top_dir` [new-function]
            -   Method which returns the top-most directory from the given path.
        -  `underline` [new-function]
            -   Append a newline to the given string with repeated characters (e.g. '-')
        -   `warn_with_traceback` [new-function]
            -   Used to override builtin `warnings.showwarning` method, will include traceback information in warning report.
    -   `log.py`
        -   `getLogger`
            -   Attached a function to new logger instances which will both log an error and raise one.  Just call `log.raise_error(msg)` on the returned `log` instance.
            -   Attached a function `log.after(msg, beg)` to report how long something took (automatically calculated).
-   math/
    +   math_core.py
        -   `argnearest` [new-function]
            +   Find the arguments in one array closest to those in another.
        -   `limit` [new-function]
            +   Limit the given value(s) to the given extrema. 
        -   `str_array` <== `strArray`
    +   statistic.py
        -   `confidence_intervals`
            +   BUG: fixed issue where multidimensional array input was leading to incorrectly shaped output arrays.
        -   `sigma`
            +   ENH: added new parameter 'boundaries' to determine whether a pair of boundaries are given for the confidence interval, or for normal behavior where the area is given.  Also added tests.
        -   `percentiles` [new-function]
            -   Function which calculates percentiles (like `np.percentile`) but with optional weighting of values.
        -   `stats_str`
            -   Changes to use local `percentiles` function instead of `np.percentile`.  Added `weights` argument, and converted from using input percentile arguments in [0, 100] range to fractions: [0.0, 1.0] range.
            -   Set `ave=False`, and remove `label` parameter.  Should be added manually on str is used from the calling code.
    +   tests/
        -   test_math_core.py
            +   `test_argnearest` [new-function]
                -   Test the new `argnearest` function.
-   plot/
    +   Hist2D.py
        -   `plot2DHist`
            +   BUG: fixed issue where grid indices were reversed -- caused errors in non-square grids.
            +   BUG: contour lines were using a different grid for some reason (unknown), was messing up edges and spacings.
            +   BUG: default `fs=None` to not change the preset font size.
        -   `plot2DHistProj`
            +   BUG: errors when x and y projection axes were turned off. 
    +   plot_core.py
        -   `colormap`
            -   ENH: added `left` and `right` parameters to allow truncation of colormaps.
        -   `cut_colormap` [new-function]
            -   ENH: new function to truncate the given colormap.
        -   `label_line` [new-function]
            +   ENH: new function to add an annotation to a given line with the appropriate placement and rotation.
        -   `plotConfFill`
            -   ENH: convert passed confidence intervals to np.array as needed.
        -   `text`
            +   ENH: Add `pad` parameter.
            +   ENH: now accepts a `loc` argument, a two-letter string which describes the location at which the text will be placed.
            +   ENH: `halign` and `valign` are now passed through the new `_parse_align()` method which will process/filter the alignment strings.  e.g. 'l' is now converted to 'left' as required for matplotlib.
        -   `setGrid`
            +   ENH: added new arguments for color and alpha.
        -   `_loc_str_to_pars`
            -   [BUG]: Was using 'lower' instead of 'bottom', triggering warning.
-   `constants.py`
    -   Added `DAY` (in seconds) variable.
-   `utils.py` [new-file]
    -   New file for general purpose, internal methods, etc.
    -   `dep_warn` [new-function]
        -   Function for sending deprecation warnings.



[0.0.8] - 2016/05/15
--------------------
-   math/
    +   math_core.py
        -   Moved many methods to new files, 'numeric.py' and 'stats.py'
        -   `around` [new-function]
            +   Round in linear or log-space, in any direction (up, down, nearest).
                This function deprecates other rounding methods
                (`ceil_log`, `floor_log`, `round_log`).
            +   When rounding in log-space, a negative value for decimals means rounding to
                an order of magnitude (in any direction).
        -   `ceil_log` [DEPRECATED] ---> `around`
        -   `floor_log` [DEPRECATED] ---> `around`
        -   `minmax`
            +   Added rounding functionality using new `around` method.
            +   Added `round_scale` parameter for interface with `around` method.
        -   `ordered_groups` [new-function]
            +   Find the locations in an array of indices which sort the input array into groups
                based on target locations.
        -   `round_log` [DEPRECATED] ---> `around`
        -   `spacing`
            +   Added `integers` parameter, if true, will create spacing in integers (linear or log)
                between the given extrema.
    +   numeric.py [new-file]
        -   Moved 'numerical' methods from 'math_core.py' to here.
    +   statistic.py [new-file]
        -   Moved 'statistical' methods from 'math_core.py' to here.
        -   `confidenceBands` [DEPRECATED] --> `confidence_bands` [new-function]
        -   `confidenceIntervals` [DEPRECATED] --> `confidence_intervals` [new-function]
        -   `sigma` [new-function]
            +   Convert from standard deviations to percentiles (inside or outside) of the normal
                distribution.
    +   tests/
        -   'test_math_core.py'
            +   Functions split off into 'test_numeric.py' and 'test_statistic.py'.
            +   Added tests for new-function `around`.
            +   Added tests for new functionality (`integers`) of `spacing()`.
        -   'test_numeric.py' [new-file]
            +   Tests for numerical functions.
        -   'test_statistic.py' [new-file]
            +   Tests for statistical functions.
            +   Tests for `sigma` function.
-   plot/
    +   Hist2D.py
        -   `plot2DHist`
            +   [MAINT] minor, allow different types of overlayed values; (see `plot2DHistProj`).
        -   `plot2DHistProj`
            +   [ENH] Allow central plot to be scatter instead of 2D histogram.
                Use `type` argument.
            +   [ENH] Add fourth subplot in the top-right corner for additional (especially
                cumulative) plots.  Still needs fine tuning, but working okay.
            +   [ENH] Add ability to overlay (write) either 'counts' or 'values' on 2D hist.
                Optional formatting available also.
            +   [ENH] Ability to plot cumulative statistics --- i.e. consider values in all bins
                (e.g.) up and to the right of the target bin, works for counts, medians, etc.
        -   `_constructFigure`
            +   [ENH] Add fourth subplot in the top-right corner, if desired.
    +   plot_core.py
        -   `backdrop`
            +   [ENH] Add option `draw` to determine if patch should be added to figure
                or only returned.
        -   `color_cycle`
            +   [ENH] Allow single `color` to be passed, from which a cycle is created by
                      using `seaborn.light_palette` or `seaborn.dark_palette`.
        -   `color_set`
            +   [ENH] Added new set of colors based on `seaborn.xkcd_palette` colors.
        -   `full_extent`
            +   [ENH] Improve to work with legends (`matplotlib.legend.Legend`).
        -   `legend`
            +   [ENH] Added `loc` parameter to automatically set x,y positions and alignment
                      based on a two-character string.
            +   [ENH] Added `mono` parameter to set font as monospaced.
        -   `strSciNot`
            +   [ENH] Added options `one` and `zero` to decide whether to include mantissa values
                      of '1.0' and whether to write '0.0' as just '0.0' (instead of 10^-inf).
        -   `test`
            +   [ENH] Now works with either `matplotlib.axes.Axes` or `matplotlib.figure.Figure`.


[0.0.7] - 2016/03/28
--------------------
-   inout/
    +   inout_core.py
        -   `ascii_table`
            +   [ENH] passing ``out = None`` will make the function return a string version of the
                table.
        -   `checkPath`
            +   [ENH] added parameter `create` to choose whether missing directories are created
                or not.
            +   [DOC] added docstrings.
        -   `iterable_notstring` [new-function]
            +   Return 'True' if the argument is an iterable and not a string type.
    +   timer.py
        -   [ENH] `Timings.report()` will return the results as a string if the parameter,
            ``out = None``.
-   math/
    +   math_core.py
        -   `_comparisonFunction` [DEPRECATED] ---> `_comparison_function` [new-function]
            +   [ENH] Returned function takes a single parameter, instead of needing the comparison
                value in each call.  Instead the comparison value is passed once to
                `_comparison_function`, just during initialization.
        -   `_comparisonFilter` [DEPRECATED] ---> `comparison_filter` [new-function]
            +   [ENH] Added options to return indices (instead of values), compare with non-zero
                comparison values, and check for finite (or not).
        -   `ceil_log` [new-function]
            +   Round up to the nearest integer in the the log10 mantissa (e.g. 23400 --> 30000)
        -   `floor_log` [new-function]
            +   Round down to the nearest integer in the the log10 mantissa (e.g. 23400 --> 20000)
        -   `frexp10`
            +   [ENH] Updated to work with negative and non-finite values.
        -   `minmax`
            +   [ENH] Extend the `prev` argument to allow for either minimum or maximum comparison
                to be `None`.
            +   [ENH] Added `limit` keyword argument to place limits on low/high extrema.
            +   [MAINT] Fully deprecated (removed) `positive`, `nonzero` keywords.
        -   `round_log` [new-function]
            +   Wrapper for `ceil_log` and `floor_log`, round in log-space in either direction.
        -   `stats_str`
            +   [ENH] Added parameter `label` to give to the output string.
    +   tests/
        -   test_math_core.py
            +   [ENH] Added *some* tests for `_comparison_function` and `_comparison_filter`.
-   plot/
    +   Hist2D.py
        -   `plot2DHist`
            +   [ENH] Added options for overplotting contour lines.  Basics work, might need some
                fine tuning.
        -   `plot2DHistProj`
            +   [ENH] added parameters to adjust the size / location of axes composing plots.
            +   [BUG] fixed issue where log-color-scales projected axes with zero values would
                fail.  Seems to be working fine.
            +   [BUG] fixed issue in right projection where the x-axis scaling would be set
                incorrectly.
            +   [BUG] fixed issue with trying to set numerous axes variables in colorbar.
            +   [ENH] updated with `cmap` and `smap` parameters passed to `plot2DHist`.
            +   [ENH] improved the way extrema are handled, especially in xprojection axis.
    +   plot_core.py
        -   `backdrop` [new-function]
            +   [ENH] Add rectangular patches behind the content of the given axes.
        -   `colormap`
            +   [ENH] Added grey colors for 'under' and 'over' (i.e. outside colormap limits).
        -   `full_extent` [new-function]
            +   [ENH] Find the bbox (or set of bbox) which contain the given axes and its contents.
        -   `legend`
            +   [BUG] fixed issue where 'center' could be repeated for `valign` and `halign`.
            +   [ENH] change the argument `fig` to be `art` -- either an axes or fig object.
            +   [ENH] added default for `handlelength` parameter; removed monospace fonts default.
        -   `line_label` [new-function]
            +   Function which draws a vertical or horizontal line, and adds an annotation to it.
        -   `plotConfFill`
            +   [ENH] Added `edges` argument to control drawing the edges of each confidence
                interval explicitly.
            +   [ENH] Added 'floor' and 'ceil' parameters to set absolute minima and maxima.
        -   `plotHistBars`
            +   [ENH] Added improved default parameters for bar plot.  Missing parameter bug fix.
        -   `plotHistLine`
            +   [ENH] Added `invert` argument to allow switching the x and y data.
        -   `position_to_extent` [new-function]
            +   [ENH] Reposition an axes object so that its 'full_extent' (see above) is at the
                intended position.
        -   `saveFigure`
            +   [ENH] check that figures saved properly.
        -   `strSciNot`
            +   [ENH] enable `None` precision --- i.e. dont show mantissa or exponent.
            +   [ENH] Updated to work with negative and non-finite values.


[0.0.6] - 2016/01/30
--------------------
-   constants.py
    +   Bug-fix where `SIGMA_T` wasn't loading properly from `astropy`.
    +   Added Electron-Scattering opacity, `KAPPA_ES`.
-   README.rst
    +   Added more information about contents and structure of package.
-   inout/
    +   inout_core.py
        -   `ascii_table` [new-function]
            +   New function which prints a table of values to the given output.
            +   Added `linewise` and `prepend` arguments, allowing the table to be printed
                line-by-line or as a single block, and for the print to be prepended with
                an additional string.
        -   `modify_exists` [new-function]
            +   Function which modifies the given filename if it already exists.  The modifications
                is appending an integer to the filename.
            +   Added tests for this function.
    +   timer.py [new-file]
        -   Provides the classes `Timer` and `Timings` which are used to time code execution and
            provided summaries of the results.  The `Timer` class is used to calculate repeated
            durations of execution for the same (type of) calculation, while the `Timings` class
            will manage the timing of many different calculations/chunks of code.
    +   tests/
        -   test_inout_core.py
            +   Fixed some issues with cleaning up (deleting) files/directories created for the
                tests.
        -   test_timer.py [new-file]
            +   Test for the classes in the new `inout/timer.py` file.  Basics tests in place.

-   math/
    +   math_core.py
        -   `groupDigitized`
            +   [Docs]: improved documentation clarifying input parameters.
        -   `stats_str` [new-function]
            +   [ENH]: Return a string with the statistics of the given array.
        -   `_comparisonFilter`
            +   [ENH]: always filter for finite values (regardless of the function arguments).
-   plot/
    +   plot_core.py
        -   `plotConfFill`
            +   [Bug]: fixed default value of `outline` which was still set to a boolean instead of
                a color string.  Caused failure when trying to save images.
        -   `colorCycle` [DEPRECATED] ---> `color_cycle` [new-function]
            +   [Docs]: added method documentation.
    +   Hist2D.py
        -   `plot2DHistProj`
            +   [ENH]: Check to make input arguments are the correct (consistent) shapes.
            +   [ENH]: Added flag 'write_counts' which overlays a string of the number of values in
                each bin of the 2D histogram.  Uses the new `counts` parameter of `plot2DHist`.
        -   `plot2DHist`
            +   [ENH]: Added parameter 'counts' for numbers to be overlaid on each bin, used by
                the `write_counts` of `plot2DHistProj`.


[0.0.5] - 2015/12/13
--------------------
-   inout/
    +   inout_core.py
        -   `dictToNPZ`
            +   Added optional `log` parameter for a ``logging.Logger`` object.
            +   Instead of raising an error for scalar parameters, cast them into arrays and
                print a warning.
    +   tests/
        -   `test_inout_core.py` [new-file]
            +   Tests for the `inout_core.py` submodule.
            +   Added tests for `npzToDict` and `dictToNPZ`.
-   math/
    +   math_core.py
        -   `confidenceBands`
            +   Added `filter` argument to select points based on how their `y` values compare to
                zero, e.g. to select for only ``y >= 0.0`` etc.
        -   `minmax`
            +   Added a `filter` argument to replace usage of `nonzero` (use `'!='`) and
                `positive` (use `'>'`).  Left both of the arguments in place, but usage of them
                will print a deprecation warning.
        -   `spacing`
            +   Updated to use `filter` argument.
-   plot/
    +   plot_core.py
        -   `plotConfFill`
            +   Added a `filter` argument to filter the values to be plotted.
            +   Added an `outline` argument to optional draw a line with a different color
                behind the median line, to make it more visible.
        -   `text`
            +   [Bug]: fixed issue where regardless of what transformation was passed, only the
                `figure` transformation was used.  Solution is to call ``plt.text`` instead of
                ``fig.text``.
    +   color2d.py [new-file]
        -   New file with classes and functions to provide color-mappings from 2D parameter spaces
            to RGB color-space.  `ScalarMappable2D` is the class which handles this mapping,
            analogous to the `matplotlib.cm.ScalarMappable` class.  Similarly, the function to
            create an instance is `zplot.color2d.colormap2d`, analogous to the
            `zcode.plot.plot_core.colormap` function.
-   constants.py
    +   Added `sigma_T` -- the Thomson-scattering cross-section in units of cm^2.


[0.0.4] - 2015/11/19
--------------------
-   General
    +   Can now run tests through python via ``>>> zcode.test()``.
-   inout/
    +   inout_core.py
        -   `mpiError` [new-method]
            +   New method to raise an error across an MPI communicator
    +   log.py
        -   `getLogger`
            +   Added the log output filename as a member variable to the newly created
                logger object.
-   math/
    +   math_core.py
        -   `argextrema` [new-method]
            +   Method to find the index of the extrema (either 'min' or 'max') with filtering
                criteria (e.g. 'ge' = filter for values ``>= 0.0``).
        -   `really1d` [new-method]
            +   Test whether a list or array is purely 1D, i.e. make sure it is not a 'jagged'
                list (or array) of lists (or arrays).
        -   `asBinEdges` [new-method]
            +   Convert a bin-specification to a list of bin-edges.  I.e. given either a set of
                bin-edges, or a number of bins (in N-dimensions), return or create those bin-edges.
        -   `confidenceIntervals` [new-method]
            +   For a pair of x and y data, bin the values by x to construct confidence intervals
                in y.
    +   tests/
        -   test_math_core.py [new-file]
            +   New location and standard for math tests using 'nose'.
            +   Moved over one of the tests for 'smooth' from previous location,
                'zcode/testing/test_Math.py' [deleted], and simplified.
-   test.sh [new-file]
    +   Bash script containing the single command to use for running nosetests.
-   testing/ [Deleted]
    +   Moved and reformatted test into new 'zcode/math/tests/test_math_core.py' file.


[0.0.3] - 2015/11/09
--------------------
-   Overall
    +   Restructured module to use subdirectories per topic (e.g. 'math') instead of single files.
    +   Implemented python3 styles into all files, with backwards compatibility.
-   CHANGES.rst [new-file]
    +   Track changes.
-   MANIFEST.in [new-file]
    +   Track files required for module.
-   version.py  [new-file]
    +   Current version information loaded from 'zcode.__init__'.
    +   Should be expanded to include git commit SHA, etc.
-   math/
    +   math_core.py
        -   Enhanced the `spline` function, and removed the secondary functions `logSpline` and
            `logSpline_resample`.  The former is included in the new functionality of `spline`,
            and the latter is too simple to warrant its own function.
        -   `strArray` [new-function]
            +   Creates a string representation of a numerical array.
        -   `indsWIthin` [new-function]
            +   Finds the indices of an array within the bounds of the given extrema.
        -   `midpoints`
            +   Enhanced to find the midpoints along an arbitrary axis.
-   plot/
    +   plot_core.py
        -   `legend` [new-method]
            +   Similar to 'text' --- just a wrapper for `matplotlib.pyplot.legend`.
        -   `plotConfFill` [new-method]
            +    Draws a median line and filled-regions for associated confidence intervals
                 (e.g. generated by `zcode.math.confidenceIntervals`).
    +   Hist2D.py
        -   Plotted histograms now use the `scipy.stats.binned_statistic` function so that more
            complicated statistics can be used.  The projected histograms are now colored to match
            the 2D main histogram.
-   inout/
    +   inout_core.py
        -   `MPI_TAGS` [new-class]
            +    A `Keys` subclass used for passing tags/status between different processors when
                 using MPI.  Commonly used in the master-slave(s) paradigm.

[0.0.2] - 2015/10/20
--------------------
